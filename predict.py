import cv2
import torch
import torch.nn.functional as F
from torch import nn
from nets.FreezeNet import FreezeNet
import numpy as np
from glob import glob
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils.util import find_ring


files=sorted(glob('datasets/*.jpg'))


model = FreezeNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model/phoneLite.pth', map_location=device))
model = nn.DataParallel(model)
model = model.to(device)
model = model.eval()
save_dir='outputs'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
input_shape=(512,512)
with tqdm(total=len(files),desc='Progress:',postfix=dict,mininterval=0.3) as pbar:
    for index,file in enumerate(files):
        imgname=file.split('/')[-1].split('.')[0]
        image = cv2.imread(file)
        h,w=image.shape[:2]
        if h<w:
            image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
            h,w=w,h
        ratio=input_shape[0]/w
        x1,x2,y1,y2,circles=find_ring(image.copy(),ratio)
        new=np.zeros((int(h),int(w),3),dtype=np.uint8)
        cv2.circle(new, (round(circles[0][0][0]/ratio), round(circles[0][0][1]/ratio)), round(circles[0][0][2]/ratio), (255, 255, 255), -1)
        image=cv2.bitwise_and(image,new)
        image=image[y1:y2,x1:x2,:]
        image=cv2.resize(image,(input_shape[0],input_shape[0]))
        img=torch.from_numpy(image/255).float().to(device)
        img=img.unsqueeze(0)
        img=img.permute(0,3,1,2)
        with torch.no_grad():
            output=model(img)
        output = output[0].permute(1,2,0).cpu().argmax(axis=-1).numpy().astype(np.uint8)
        new = cv2.bitwise_and(image,image,mask=output)
        cv2.imwrite('{}/{}'.format(save_dir,os.path.basename(file)),new)
        pbar.update(1)
    