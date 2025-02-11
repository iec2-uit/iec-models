from glob import glob
import imp
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import timm
import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
import pydicom
#from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
import requests
import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np
import json
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import albumentations 
from albumentations.pytorch import ToTensorV2

from albumentations import (
          HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
          Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
          IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
          IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
      )

class IEC(Dataset):
    CFG = {
        'fold_num': 5,
        'seed': 719,
        'model_arch': 'levit_256',
        'img_size': 224,
        'epochs': 1,
        'train_bs': 9,
        'valid_bs': 9,
        'T_0': 10,
        'lr': 1e-4,
        'min_lr': 1e-6,
        'weight_decay':1e-6,
        'num_workers': 4,
        'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
        'verbose_step': 1,
        'device': 'cuda:0'
      #  'device': 'cpu'
    }
    def rand_bbox(size, lam):
      W = size[0]
      H = size[1]
      cut_rat = np.sqrt(1. - lam)
      cut_w = np.int(W * cut_rat)
      cut_h = np.int(H * cut_rat)

      # uniform
      cx = np.random.randint(W)
      cy = np.random.randint(H)

      bbx1 = np.clip(cx - cut_w // 2, 0, W)
      bby1 = np.clip(cy - cut_h // 2, 0, H)
      bbx2 = np.clip(cx + cut_w // 2, 0, W)
      bby2 = np.clip(cy + cut_h // 2, 0, H)
      return bbx1, bby1, bbx2, bby2

      
    def get_train_transforms():
        return Compose([
                RandomResizedCrop(IEC.CFG['img_size'], IEC.CFG['img_size']),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
      
            
    def get_valid_transforms():
        return Compose([
                CenterCrop(IEC.CFG['img_size'], IEC.CFG['img_size'], p=1.),
                Resize(IEC.CFG['img_size'], IEC.CFG['img_size']),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
        
    def __init__(self, df, data_root,
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (CFG['img_size'], CFG['img_size']),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 },
                

                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.output_label = output_label
        self.one_hot_label = one_hot_label
    
        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
           target = self.labels[index]

        #print("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        img  = IEC.get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            
            img = self.transforms(image=img)['image']

        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)
                
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
    
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = IEC.get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                
                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                #print(mask.shape)

                #assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum()/IEC.CFG['img_size']/IEC.CFG['img_size']
                target = rate*target + (1.-rate)*self.labels[fmix_ix]
                #print(target, mask, img)
                #assert False
        
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = IEC.get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                print(cmix_img)
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = IEC.rand_bbox((IEC.CFG['img_size'], IEC.CFG['img_size']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (IEC.CFG['img_size'] * IEC.CFG['img_size']))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
                
            #print('-', img.sum())
            #print(target)
            #assert False
                            
        # do label smoothing
        #print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img

    def download(name = 'errors'):
            if(name == 'errors'):
                return
            else:
                r = requests.get("http://149.28.146.58:3000/datasets/link")
                data = r.json()
                for d in data:
                    if(d['name'] == name):
                        link = d['link']
                url =  link
                local_filename = url.split('/')[-1]
                print('Starting Download, Please wait.....')
                chunk_size = 4096
                r = requests.get(url, stream=True)
                with open(local_filename, 'wb') as f:
                   print("Downloading %s" % local_filename)
                   response = requests.get(url, stream=True)
                   total_length = response.headers.get('content-length')
                   if total_length is None: # no content length header
                      f.write(response.content)
                   else:
                        dl = 0
                        total_length = int(total_length)
                        for data in response.iter_content(int(total_length / 100)):
                            dl += len(data)
                            f.write(data)
                            done = int(50 * dl / total_length)
                            sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done))  )
                            sys.stdout.flush()

                print('\nDownload '+ name +' complete.')
                with ZipFile(local_filename, 'r') as zip:
                    print('Extracting all the files now...')
                    zip.extractall('./model_data/')
                # train = pd.read_csv('/content/iec-models/model_data/train.csv')
                return local_filename

    def prepare_dataloader(df, trn_idx, val_idx, data_root='/content/iec-models/model_data/train_images'):
    
        from catalyst.data.sampler import BalanceClassSampler
    
        train_ = df.loc[trn_idx,:].reset_index(drop=True)
        valid_ = df.loc[val_idx,:].reset_index(drop=True)
            
        train_ds = IEC(train_, data_root, transforms=IEC.get_train_transforms(), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
        valid_ds = IEC(valid_, data_root, transforms=IEC.get_valid_transforms(), output_label=True)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=IEC.CFG['train_bs'],
            pin_memory=False,
            drop_last=False,
            shuffle=True,        
            num_workers=IEC.CFG['num_workers'],
            #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
        )
        val_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=IEC.CFG['valid_bs'],
            num_workers=IEC.CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )
        return train_loader, val_loader

    def train_one_epoch(fold, epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
        model.train()

        t = time.time()
        running_loss = None

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()
            # print(imgs,image_labels)
            scaler = GradScaler()
            with autocast():
          
                image_preds = model(imgs)   #output = model(input)

                loss = loss_fn(image_preds[0], image_labels)
                
                scaler.scale(loss).backward()

                if running_loss is None:
                    running_loss = loss.item()
                else:
                    running_loss = running_loss * .99 + loss.item() * .01

                if ((step + 1) %  IEC.CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                    # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() 
                    
                    if scheduler is not None and schd_batch_update:
                        scheduler.step()

                if ((step + 1) % IEC.CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                    description = f'epoch {epoch} loss: {running_loss:.4f}'
                    
                    pbar.set_description(description)
                  
        if scheduler is not None and not schd_batch_update:
           scheduler.step()
            
    def valid_one_epoch(isTrain, fold, epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
        model.eval()

        t = time.time()
        loss_sum = 0
        sample_num = 0
        image_preds_all = []
        image_targets_all = []
        
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()
            
            image_preds = model(imgs)   #output = model(input)
            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
            image_targets_all += [image_labels.detach().cpu().numpy()]
            
            loss = loss_fn(image_preds, image_labels)
            
            loss_sum += loss.item()*image_labels.shape[0]
            sample_num += image_labels.shape[0]  

            if ((step + 1) % IEC.CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
                description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
                pbar.set_description(description)
        
        image_preds_all = np.concatenate(image_preds_all)
        image_targets_all = np.concatenate(image_targets_all)
        print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
        print ("Classification report: ", (classification_report(image_targets_all, image_preds_all)))
        print ("F1 micro averaging:",(f1_score(image_targets_all, image_preds_all, average='micro')))


        if isTrain == 1:
          writer.add_scalar("Training loss", loss_sum/sample_num, epoch + fold*33)
          writer.add_scalar("Training accurssacy",(image_preds_all==image_targets_all).mean(), epoch + fold*33)
        if isTrain == 0:
          writer.add_scalar("Validation loss", loss_sum/sample_num, epoch + fold*33)
          writer.add_scalar("Validation accuracy",(image_preds_all==image_targets_all).mean(), epoch + fold*33)

        
        if scheduler is not None:
            if schd_loss_update:
                scheduler.step(loss_sum/sample_num)
            else:
                scheduler.step()
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    
    def get_img(path = '/content/iec-models/model_data/train_images/'):
        im_bgr = cv2.imread(path)
        im_rgb = im_bgr[:, :, [2, 1, 0]]
        return im_rgb

    def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
   
      assert y_true.ndim == 1
      assert y_pred.ndim == 1 or y_pred.ndim == 2
      
      if y_pred.ndim == 2:
          y_pred = y_pred.argmax(dim=1)
      
      print(y_true)
      print(y_pred)
      tp = (y_true * y_pred).sum().astype(float)
      tn = ((1 - y_true) * (1 - y_pred)).sum().astype(float)
      fp = ((1 - y_true) * y_pred).sum().astype(float)
      fn = (y_true * (1 - y_pred)).sum().astype(float)
      
      epsilon = 1e-7
      
      precision = tp / (tp + fp + epsilon)
      recall = tp / (tp + fn + epsilon)
      
      f1 = 2* (precision*recall) / (precision + recall + epsilon)
      #f1.requires_grad = is_training
      return f1
