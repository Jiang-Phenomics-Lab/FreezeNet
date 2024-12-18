import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from functools import partial
import math
import numpy as np




def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def compute_total_loss(coarse_output, labels, selected_points_list, refined_output_at_points_list, alpha=1.0, beta=1.0):
    CE_loss = F.cross_entropy(coarse_output, labels)
    dice_loss= Dice_loss(coarse_output, labels)
    segmentation_loss=CE_loss+dice_loss
    pointrend_loss = 0
    for b in range(len(selected_points_list)):
        gt_at_selected_points = labels[b, selected_points_list[b][:, 0], selected_points_list[b][:, 1]]
        pointrend_loss += F.cross_entropy(refined_output_at_points_list[b], gt_at_selected_points.unsqueeze(0))
    pointrend_loss /= len(selected_points_list)
    total_loss = alpha * segmentation_loss + beta * pointrend_loss
    return total_loss


def compute_accuracy(pred, target):
    pred = F.softmax(pred, dim=1)  
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()


def compute_f1_score(pred, target, num_classes=2):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    f1_score = 0.0
    for i in range(num_classes):
        true_positives = ((pred == i) & (target == i)).float().sum()
        false_positives = ((pred == i) & (target != i)).float().sum()
        false_negatives = ((pred != i) & (target == i)).float().sum()
        
        precision = true_positives / (true_positives + false_positives + 1e-5)
        recall = true_positives / (true_positives + false_negatives + 1e-5)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
        f1_score += f1
    
    f1_score /= num_classes
    return f1_score.item()


def setup_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    
def warm_up_cos(lr, min_lr, total_iters):
    def warm_cos(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (1.0+ math.cos(math.pi* (iters - warmup_total_iters)/ (total_iters - warmup_total_iters - no_aug_iter)))
        return lr


    warmup_total_iters  = min(max(0.1 * total_iters, 1), 3)
    warmup_lr_start     = max(0.1 * lr, 1e-6)
    no_aug_iter         = min(max(0.3 * total_iters, 1), 15)

    func = partial(warm_cos ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    return func


def find_ring(temp,ratio):
    h,w=temp.shape[:2]
    temp=cv2.resize(temp,(int(w*ratio),int(h*ratio)))
    gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
    b,g,r=cv2.split(temp)
    crt1=abs(gray.astype(np.int32)-r.astype(np.int32))<50
    crt2=abs(gray.astype(np.int32)-g.astype(np.int32))<50
    crt3=abs(gray.astype(np.int32)-b.astype(np.int32))<50
    rcrt1=r.astype(np.int32)-b.astype(np.int32)>15
    rcrt2=g.astype(np.int32)-b.astype(np.int32)>15
    rcrt3=abs(r.astype(np.int32)-g.astype(np.int32))>20
    rcrt4=cv2.bitwise_and(rcrt1.astype(np.uint8),rcrt2.astype(np.uint8))
    rcrt=cv2.bitwise_and(rcrt3.astype(np.uint8),rcrt4.astype(np.uint8))
    crt4=gray>100
    crt5=cv2.bitwise_and(crt1.astype(np.uint8),crt2.astype(np.uint8))
    crt6=cv2.bitwise_and(crt3.astype(np.uint8),crt5.astype(np.uint8))
    crt7=cv2.bitwise_and(crt6.astype(np.uint8),1-rcrt2.astype(np.uint8))
    mask=cv2.bitwise_and(crt4.astype(np.uint8),crt7.astype(np.uint8))
    temp = cv2.bitwise_and(temp, temp, mask=mask)
    gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
    kernel_size = (5, 5) 
    gray = cv2.blur(gray, kernel_size)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 500, param1=40, param2=80, minRadius=200, maxRadius=240)
    x1,x2,y1,y2=(round((circles[0][0][0]-circles[0][0][2])/ratio),
             round((circles[0][0][0]+circles[0][0][2])/ratio),
             round((circles[0][0][1]-circles[0][0][2])/ratio),
             round((circles[0][0][1]+circles[0][0][2])/ratio))
    y1=max(y1,0)
    x1=max(x1,0)
    y2=min(y2,h)
    x2=min(x2,w)
    return x1,x2,y1,y2,circles


if __name__=='__main__':
    pred = torch.randn(8, 2, 256, 256)  
    labels = torch.randint(0, 2, (8, 256, 256), dtype=torch.long) 
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, labels)
    dice_loss=Dice_loss(pred, labels)
    acc=compute_accuracy(pred, labels)
    f1_score=compute_f1_score(pred, labels,2)
    print(loss,dice_loss,acc,f1_score)