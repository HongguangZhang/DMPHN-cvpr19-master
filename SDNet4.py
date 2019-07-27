import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import GoProDataset

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 3000)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 2)
parser.add_argument("-c","--cropsize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

#Hyper Parameters
METHOD = "SDNet4"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
CROP_SIZE = args.cropsize

def save_deblur_images(images, iteration, epoch):
    filename = './checkpoints/' + METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + "_deblur.png"
    torchvision.utils.save_image(images, filename)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(0.5 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    print("init data folders")

    encoder = {}
    decoder = {}
    encoder_optim = {}
    decoder_optim = {}
    encoder_scheduler = {}
    decoder_scheduler = {}
    for s in ['s1', 's2', 's3', 's4']:
        encoder[s] = {}
        decoder[s] = {}
        encoder_optim[s] = {}
        decoder_optim[s] = {}
        encoder_scheduler[s] = {}
        decoder_scheduler[s] = {}
        for lv in ['lv1', 'lv2', 'lv3']:
            encoder[s][lv] = models.Encoder()
            decoder[s][lv] = models.Decoder()
            encoder[s][lv].apply(weight_init).cuda(GPU)
            decoder[s][lv].apply(weight_init).cuda(GPU)
            encoder_optim[s][lv] = torch.optim.Adam(encoder[s][lv].parameters(),lr=LEARNING_RATE)
            encoder_scheduler[s][lv] = StepLR(encoder_optim[s][lv],step_size=1000,gamma=0.1)
            decoder_optim[s][lv] = torch.optim.Adam(decoder[s][lv].parameters(),lr=LEARNING_RATE)
            decoder_scheduler[s][lv] = StepLR(decoder_optim[s][lv],step_size=1000,gamma=0.1)
            if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_" + s + "_" + lv + ".pkl")):
                encoder[s][lv].load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_" + s + "_" + lv + ".pkl")))
                print("load encoder_" + s + "_" + lv + " successfully!")
            if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_" + s + "_" + lv + ".pkl")):
                decoder[s][lv].load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_" + s + "_" + lv + ".pkl")))
                print("load decoder_" + s + "_" + lv + " successfully!")

    
    if os.path.exists('./checkpoints/' + METHOD) == False:
        os.system('mkdir ./checkpoints/' + METHOD)    
            
    for epoch in range(args.start_epoch, EPOCHS):
        for s in ['s1', 's2', 's3', 's4']:
            for lv in ['lv1', 'lv2', 'lv3']:
                encoder_scheduler[s][lv].step(epoch)
                decoder_scheduler[s][lv].step(epoch)      
        
        print("Training...")
        
        train_dataset = GoProDataset(
            blur_image_files = './datas/GoPro/train_blur_file.txt',
            sharp_image_files = './datas/GoPro/train_sharp_file.txt',
            root_dir = './datas/GoPro/',
            crop = True,
            crop_size = CROP_SIZE,
            rotation = True,
            color_augment = True,                        
            transform = transforms.Compose([           
                transforms.ToTensor()
                ]))
        train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
        
        for iteration, inputs in enumerate(train_dataloader):            
            mse = nn.MSELoss().cuda(GPU)
            images = {}
            feature = {}
            residual = {}
            for s in ['s1', 's2', 's3', 's4']:
                feature[s] = {}
                residual[s] = {}
				            
            images['gt'] = Variable(inputs['sharp_image'] - 0.5).cuda(GPU)
            images['lv1'] = Variable(inputs['blur_image'] - 0.5).cuda(GPU)           
            H = images['lv1'].size(2)
            W = images['lv1'].size(3)
            
            images['lv2_1'] = images['lv1'][:,:,0:int(H/2),:]
            images['lv2_2'] = images['lv1'][:,:,int(H/2):H,:]
            images['lv3_1'] = images['lv2_1'][:,:,:,0:int(W/2)]
            images['lv3_2'] = images['lv2_1'][:,:,:,int(W/2):W]
            images['lv3_3'] = images['lv2_2'][:,:,:,0:int(W/2)]
            images['lv3_4'] = images['lv2_2'][:,:,:,int(W/2):W]

            s = 's1'		
            feature[s]['lv3_1'] = encoder[s]['lv3'](images['lv3_1'])
            feature[s]['lv3_2'] = encoder[s]['lv3'](images['lv3_2'])
            feature[s]['lv3_3'] = encoder[s]['lv3'](images['lv3_3'])
            feature[s]['lv3_4'] = encoder[s]['lv3'](images['lv3_4'])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3)
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3)
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](images['lv2_1'] + residual[s]['lv3_top']) + feature[s]['lv3_top']
            feature[s]['lv2_2'] = encoder[s]['lv2'](images['lv2_2'] + residual[s]['lv3_bot']) + feature[s]['lv3_bot']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])

            feature[s]['lv1'] = encoder[s]['lv1'](images['lv1'] + residual[s]['lv2']) + feature[s]['lv2']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])
            
            s = 's2'
            ps = 's1'
            feature[s]['lv3_1'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),0:int(W/2)])
            feature[s]['lv3_2'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),int(W/2):W])
            feature[s]['lv3_3'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,0:int(W/2)])
            feature[s]['lv3_4'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,int(W/2):W])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,0:int(H/2),:] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + feature[ps]['lv2_1']
            feature[s]['lv2_2'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,int(H/2):H,:] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + feature[ps]['lv2_2']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2']) + residual['s1']['lv1']

            feature[s]['lv1'] = encoder[s]['lv1'](residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + feature[ps]['lv1']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

            s = 's3'
            ps = 's2'
            feature[s]['lv3_1'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),0:int(W/2)])
            feature[s]['lv3_2'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),int(W/2):W])
            feature[s]['lv3_3'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,0:int(W/2)])
            feature[s]['lv3_4'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,int(W/2):W])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,0:int(H/2),:] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + feature[ps]['lv2_1']
            feature[s]['lv2_2'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,int(H/2):H,:] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + feature[ps]['lv2_2']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2']) + residual['s1']['lv1']

            feature[s]['lv1'] = encoder[s]['lv1'](residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + feature[ps]['lv1']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])


            s = 's4'
            ps = 's3'
            feature[s]['lv3_1'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),0:int(W/2)])
            feature[s]['lv3_2'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),int(W/2):W])
            feature[s]['lv3_3'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,0:int(W/2)])
            feature[s]['lv3_4'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,int(W/2):W])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,0:int(H/2),:] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + feature[ps]['lv2_1']
            feature[s]['lv2_2'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,int(H/2):H,:] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + feature[ps]['lv2_2']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2']) + residual['s1']['lv1']

            feature[s]['lv1'] = encoder[s]['lv1'](residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + feature[ps]['lv1']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])
                
            loss = mse(residual['s4']['lv1'], images['gt']) + mse(residual['s3']['lv1'], images['gt']) +  mse(residual['s2']['lv1'], images['gt']) + mse(residual['s1']['lv1'], images['gt'])
            
            for s in ['s1', 's2', 's3', 's4']:
            	for lv in ['lv1', 'lv2', 'lv3']:
            	    encoder[s][lv].zero_grad()
            	    decoder[s][lv].zero_grad()
            		
            loss.backward()
            		
            for s in ['s1', 's2', 's3', 's4']:
            	for lv in ['lv1', 'lv2', 'lv3']:
            	    encoder_optim[s][lv].step()
            	    decoder_optim[s][lv].step()	               		    
            
            if (iteration+1)%10 == 0:
                print(METHOD + "   epoch:", epoch, "iteration:", iteration+1, "loss:",loss.data[0])
                
        if (epoch)%200==0:
            if os.path.exists('./checkpoints/' + METHOD + '/epoch' + str(epoch)) == False:
            	os.system('mkdir ./checkpoints/' + METHOD + '/epoch' + str(epoch))
            	
            for s in ['s1', 's2', 's3', 's4']:
                for lv in ['lv1', 'lv2', 'lv3']:
                    torch.save(encoder[s][lv].state_dict(),str('./checkpoints/' + METHOD + '/epoch' + str(epoch) + "/encoder_" + s + "_" + lv + ".pkl"))
                    torch.save(decoder[s][lv].state_dict(),str('./checkpoints/' + METHOD + '/epoch' + str(epoch) + "/decoder_" + s + "_" + lv + ".pkl"))
            
            print("Testing...")
            test_dataset = GoProDataset(
                blur_image_files = './datas/GoPro/test_blur_file.txt',
                sharp_image_files = './datas/GoPro/test_sharp_file.txt',
                root_dir = './datas/GoPro/',
                transform = transforms.Compose([
                    transforms.ToTensor()
                ]))
            test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)
        
            for iteration, inputs in enumerate(test_dataloader):
                with torch.no_grad():
                    images['lv1'] = Variable(inputs['blur_image'] - 0.5).cuda(GPU)
                    H = images['lv1'].size(2)
                    W = images['lv1'].size(3)
                    images['lv2_1'] = images['lv1'][:,:,0:int(H/2),:]
                    images['lv2_2'] = images['lv1'][:,:,int(H/2):H,:]
                    images['lv3_1'] = images['lv2_1'][:,:,:,0:int(W/2)]
                    images['lv3_2'] = images['lv2_1'][:,:,:,int(W/2):W]
                    images['lv3_3'] = images['lv2_2'][:,:,:,0:int(W/2)]
                    images['lv3_4'] = images['lv2_2'][:,:,:,int(W/2):W]

                    s = 's1'		
                    feature[s]['lv3_1'] = encoder[s]['lv3'](images['lv3_1'])
                    feature[s]['lv3_2'] = encoder[s]['lv3'](images['lv3_2'])
                    feature[s]['lv3_3'] = encoder[s]['lv3'](images['lv3_3'])
                    feature[s]['lv3_4'] = encoder[s]['lv3'](images['lv3_4'])
                    feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3)
                    feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3)
                    residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
                    residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

                    feature[s]['lv2_1'] = encoder[s]['lv2'](images['lv2_1'] + residual[s]['lv3_top']) + feature[s]['lv3_top']
                    feature[s]['lv2_2'] = encoder[s]['lv2'](images['lv2_2'] + residual[s]['lv3_bot']) + feature[s]['lv3_bot']
                    feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
                    residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])
                    
                    feature[s]['lv1'] = encoder[s]['lv1'](images['lv1'] + residual[s]['lv2']) + feature[s]['lv2']
                    residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])
            		
                    s = 's2'
                    ps = 's1'
                    feature[s]['lv3_1'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),0:int(W/2)])
                    feature[s]['lv3_2'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),int(W/2):W])
                    feature[s]['lv3_3'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,0:int(W/2)])
                    feature[s]['lv3_4'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,int(W/2):W])
                    feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
                    feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
                    residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
                    residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

                    feature[s]['lv2_1'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,0:int(H/2),:] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + feature[ps]['lv2_1']
                    feature[s]['lv2_2'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,int(H/2):H,:] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + feature[ps]['lv2_2']
                    feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
                    residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])

                    feature[s]['lv1'] = encoder[s]['lv1'](residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + feature[ps]['lv1']
                    residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])
                    
                    s = 's3'
                    ps = 's2'
                    feature[s]['lv3_1'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),0:int(W/2)])
                    feature[s]['lv3_2'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),int(W/2):W])
                    feature[s]['lv3_3'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,0:int(W/2)])
                    feature[s]['lv3_4'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,int(W/2):W])
                    feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
                    feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
                    residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
                    residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

                    feature[s]['lv2_1'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,0:int(H/2),:] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + feature[ps]['lv2_1']
                    feature[s]['lv2_2'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,int(H/2):H,:] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + feature[ps]['lv2_2']
                    feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
                    residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])

                    feature[s]['lv1'] = encoder[s]['lv1'](residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + feature[ps]['lv1']
                    residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

                    s = 's4'
                    ps = 's3'
                    feature[s]['lv3_1'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),0:int(W/2)])
                    feature[s]['lv3_2'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),int(W/2):W])
                    feature[s]['lv3_3'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,0:int(W/2)])
                    feature[s]['lv3_4'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,int(W/2):W])
                    feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
                    feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
                    residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
                    residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

                    feature[s]['lv2_1'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,0:int(H/2),:] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + feature[ps]['lv2_1']
                    feature[s]['lv2_2'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,int(H/2):H,:] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + feature[ps]['lv2_2']
                    feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
                    residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])

                    feature[s]['lv1'] = encoder[s]['lv1'](residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + feature[ps]['lv1']
                    residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])
                                        
                    deblurred_image = residual[s]['lv1']

                    save_deblur_images(deblurred_image.data + 0.5, 4, iteration, epoch)

        for s in ['s1', 's2', 's3', 's4']:
            for lv in ['lv1', 'lv2', 'lv3']:
                torch.save(encoder[s][lv].state_dict(),str('./checkpoints/' + METHOD + "/encoder_" + s + "_" + lv + ".pkl"))
                torch.save(decoder[s][lv].state_dict(),str('./checkpoints/' + METHOD + "/decoder_" + s + "_" + lv + ".pkl"))
                

if __name__ == '__main__':
    main()

        

        

