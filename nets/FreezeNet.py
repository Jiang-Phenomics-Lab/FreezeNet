import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

        
class CrossAttentionModule(nn.Module):
    def __init__(self, channels1, channels2,outc):
        super(CrossAttentionModule, self).__init__()        
        self.conv1= nn.Sequential(
                                    DepthwiseSeparableConv2d(channels1+channels2,(channels1+channels2)//2,3,1,1),
                                    nn.BatchNorm2d((channels1+channels2)//2)
                                 )
        self.conv2= nn.Sequential(
                                  DepthwiseSeparableConv2d((channels1+channels2)//2,(channels1+channels2)//4,3,1,1),
                                  nn.BatchNorm2d((channels1+channels2)//4),
                                  nn.ReLU(inplace=True),
                                  DepthwiseSeparableConv2d((channels1+channels2)//4,1,3,1,1),
                                  nn.BatchNorm2d(1)
                                 )
        self.conv3= nn.Sequential(
                                  DepthwiseSeparableConv2d((channels1+channels2)//2,(channels1+channels2)//4,3,1,1),
                                  nn.BatchNorm2d((channels1+channels2)//4),
                                  nn.ReLU(inplace=True),
                                  DepthwiseSeparableConv2d((channels1+channels2)//4,1,3,1,1),
                                  nn.BatchNorm2d(1)
            
                                 )
        
        
    def forward(self, A, B):
        feature = torch.cat([A, B], dim=1)
        feature= self.conv1(feature)
        feature= F.relu(feature)
        attention_A = torch.sigmoid(self.conv2(feature))
        attention_B = torch.sigmoid(self.conv3(feature))
        weighted_A = A * attention_A
        weighted_B = B * attention_B
        
        weights= torch.cat((weighted_A, weighted_B),dim=1)
        self.attention_weights = (attention_A, attention_B)
        self.addw_weights = (weighted_A, weighted_B)
        return weights
    
    
class doubleConv(nn.Module):
    def __init__(self, channels1, channels2,outc):
        super(doubleConv, self).__init__()
        self.conv1= nn.Sequential(
                      DepthwiseSeparableConv2d(channels1+channels2,(channels1+channels2)//2,3,1,1),
                      nn.BatchNorm2d((channels1+channels2)//2),
                    )
        self.conv2= nn.Sequential(
                     InvertedResidual((channels1+channels2)//2,outc,1,1.5),
                     nn.BatchNorm2d(outc),
                     nn.ReLU(inplace=True),
                     InvertedResidual(outc,outc,1,1.5),
                     nn.BatchNorm2d(outc)
                    )
    def forward(self, x):
        weights= self.conv1(x)
        weights= F.relu(weights)
        weights= self.conv2(weights)
        weights= F.relu(weights)
        return weights
    

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class FreezeNet(nn.Module):
    def __init__(self, width_mult=1.,num_classes=2):
        super(FreezeNet, self).__init__()
        
        self.cfgs = [
            [1,  16, 1, 2],
            [6,  32, 2, 2],
            [6,  64, 3, 2],
            [6,  128, 4, 2],
        ]
        
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        #self.conv1=conv_3x3_bn(3, input_channel, 2)

        self.features:List[nn.Module] = []

        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                if t==1:
                    input_channel=3
                self.features.append(InvertedResidual(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features=nn.Sequential(*self.features)
        
        
        
        self.attention1=CrossAttentionModule(16,32,16)
        self.attention2=CrossAttentionModule(32,64,32)
        self.attention3=CrossAttentionModule(64,128,64)
        
        self.attention4=CrossAttentionModule(16,32,16)
        self.attention5=CrossAttentionModule(32,64,32)
        self.attention6=CrossAttentionModule(16,32,16)
        
        self.cat_conv1=doubleConv(16,32,16)
        self.cat_conv2=doubleConv(32,64,32)
        self.cat_conv3=doubleConv(64,128,64)
        
        self.cat_conv4=doubleConv(16,32,16)
        self.cat_conv5=doubleConv(32,64,32)
        self.cat_conv6=doubleConv(16,32,16)
        
        self.final = nn.Conv2d(16,num_classes, 1)
        self._initialize_weights()

    def forward(self, x):
        input_shape=x.shape[-1]
        
        outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in ['0', '2', '5', '9']:
                outputs.append(x)
                
        x1_1,x1_2,x1_3,x1_4 = outputs

        x2_1=F.interpolate(x1_2, size=input_shape//2, mode='bilinear', align_corners=False)
        x2_1=self.attention1(x1_1,x2_1)
        x2_1=self.cat_conv1(x2_1)
        
        x2_2=F.interpolate(x1_3, size=input_shape//4, mode='bilinear', align_corners=False)
        x2_2=self.attention2(x1_2,x2_2)
        x2_2=self.cat_conv2(x2_2)
        
        x2_3=F.interpolate(x1_4, size=input_shape//8, mode='bilinear', align_corners=False)
        x2_3=self.attention3(x1_3,x2_3)
        x2_3=self.cat_conv3(x2_3)
        
        x3_1=F.interpolate(x2_2, size=input_shape//2, mode='bilinear', align_corners=False)
        x3_1=self.attention4(x2_1,x3_1)
        x3_1=self.cat_conv4(x3_1)
        
        x3_2=F.interpolate(x2_3, size=input_shape//4, mode='bilinear', align_corners=False)
        x3_2=self.attention5(x2_2,x3_2)
        x3_2=self.cat_conv5(x3_2)
        
        x4_1=F.interpolate(x3_2, size=input_shape//2, mode='bilinear', align_corners=False)
        x4_1=self.attention6(x3_1,x4_1)
        x4_1=self.cat_conv6(x4_1)
        
        x5_1=F.interpolate(x4_1, size=input_shape, mode='bilinear', align_corners=False)
        x=self.final(x5_1)
        x=F.relu(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
