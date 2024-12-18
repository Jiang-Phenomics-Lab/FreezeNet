from nets.FreezeNet import FreezeNet
import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
import random
from tqdm import tqdm
from utils.util import *
from utils.dataloader import *


input_shape=(512,512)
num_classes=2
momentum= 0.9
weight_decay= 5e-4  
batch_size=4
val_batch_size=4
CUDA=True
Epoch=4000
log_dir='FreezyNet'
log_dir='logs/'+log_dir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
Init_lr= 1e-4  
Min_lr= 1e-6


files=sorted(glob('../datasets/labels/*.jpg')) #change your own dataset path (labels), more info in data FreezeNetDataset
random.seed(6666)
random.shuffle(files)
files=files[:121]
validation=0.1
val_lines=files[:int(validation*len(files))]
train_lines=files[int(validation*len(files)):]


train_dataset = FreezeNetDataset(train_lines, input_shape, num_classes, True)
val_dataset = FreezeNetDataset(val_lines, input_shape, num_classes, False)

print(len(files))



gen = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 2, pin_memory=True,
                            drop_last=True, collate_fn=freezenet_dataset_collate)
gen_val= DataLoader(val_dataset  , shuffle = True, batch_size = val_batch_size, num_workers = 2, pin_memory=True,
                            drop_last=True, collate_fn=freezenet_dataset_collate)


epoch_step = len(train_dataset) // batch_size
epoch_step_val = len(val_dataset) // val_batch_size


#model
model=FreezeNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if CUDA:
    cudnn.benchmark = True
    model=model.to(device)
    
criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer=optim.Adam(model.parameters(), Init_lr, betas = (momentum, 0.999), weight_decay = weight_decay)
warmup_cos_lr_func = warm_up_cos(Init_lr, Min_lr, Epoch)

record='epoch,loss,acc,f1_score,val loss,val acc,val f1_score\n'
for epoch in range(Epoch):
    total_loss= 0
    total_accuracy  = 0
    total_f1_score  = 0
    val_total_loss= 0
    val_total_accuracy  = 0
    val_total_f1_score  = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = warmup_cos_lr_func(epoch)
    model.train()
    with tqdm(total=epoch_step,desc='Epoch {}/{}'.format(epoch + 1,Epoch),postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs,labels,onehot_labels=batch
            with torch.no_grad():
                if CUDA:
                    imgs=imgs.cuda()
                    labels=labels.cuda()
                    onehot_labels=onehot_labels.cuda()

            optimizer.zero_grad()
            output = model(imgs)
            loss=F.cross_entropy(output, labels)+Dice_loss(output, onehot_labels)

            with torch.no_grad():
                total_loss+=loss.item()
                total_accuracy+=compute_accuracy(output,labels)
                total_f1_score+=compute_f1_score(output,labels)

            loss.backward()
            optimizer.step()
            pbar.set_postfix(**{'acc':total_accuracy/(iteration + 1),
                            'f1_score':total_f1_score/(iteration + 1),
                                'total_loss': total_loss / (iteration + 1),
                            'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)


    model.eval()

    with tqdm(total=epoch_step_val,desc='Epoch {}/{}'.format(epoch + 1,Epoch),postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs,labels,onehot_labels=batch
            with torch.no_grad():
                if CUDA:
                    imgs=imgs.cuda()
                    labels=labels.cuda()
                    onehot_labels=onehot_labels.cuda()

                output = model(imgs)

                loss=F.cross_entropy(output, labels)+Dice_loss(output, onehot_labels)

                val_total_loss+=loss.item()
                val_total_accuracy+=compute_accuracy(output,labels)
                val_total_f1_score+=compute_f1_score(output,labels)


            pbar.set_postfix(**{'acc':val_total_accuracy/(iteration + 1),
                            'f1_score':val_total_f1_score/(iteration + 1),
                                'total_loss': val_total_loss / (iteration + 1),
                            'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    print('Epoch {} finished'.format(epoch+1))

    if epoch==0:
        torch.save(model.state_dict(),'{}/model.pth'.format(log_dir))
        
        min_val_loss=val_total_loss / epoch_step_val
        min_val_f1_score=val_total_f1_score / epoch_step_val
        
        record='{},{},{},{},{},{},{}\n'.format(epoch+1,total_loss / epoch_step,total_accuracy / epoch_step, total_f1_score / epoch_step,
                                val_total_loss / epoch_step_val,val_total_accuracy / epoch_step_val,  val_total_f1_score / epoch_step_val)
        with open('{}/log.txt'.format(log_dir),'w') as f:
            f.write(record)
    else:
        record='{},{},{},{},{},{},{}\n'.format(epoch+1,total_loss / epoch_step,total_accuracy / epoch_step, total_f1_score / epoch_step,
                                val_total_loss / epoch_step_val,val_total_accuracy / epoch_step_val,  val_total_f1_score / epoch_step_val)
        with open('{}/log.txt'.format(log_dir),'a') as f:
            f.write(record)
        if min_val_loss>val_total_loss / epoch_step_val:
            torch.save(model.state_dict(),'{}/model.pth'.format(log_dir))
            min_val_loss=val_total_loss / epoch_step_val
            print('model has been saved')
            
        if min_val_f1_score<val_total_f1_score / epoch_step_val:
            torch.save(model.state_dict(),'{}/model_f1-score.pth'.format(log_dir))
            min_val_f1_score=val_total_f1_score / epoch_step_val
            print('model(f1-score) has been saved')
        if min_val_f1_score<val_total_f1_score / epoch_step_val and min_val_f1_score<val_total_f1_score / epoch_step_val:
            torch.save(model.state_dict(),'{}/model_all.pth'.format(log_dir))
            print('model(all) has been saved')
            
