from iec import IEC
from CassvaImgClassifier import CassvaImgClassifier
import sklearn
from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch.quantization
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch import nn


if __name__ == '__main__':
     # for training only, need nightly build pytorch
    IEC.download()
    train = pd.read_csv('/content/iec-models/model_data/train.csv')

    IEC.seed_everything(IEC.CFG['seed'])
    
    IEC.folds = StratifiedKFold(n_splits=IEC.CFG['fold_num'], shuffle=True, random_state=IEC.CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    device = torch.device(IEC.CFG['device'])
    print(device)
        
    model = torch.hub.load('facebookresearch/LeViT:main', 'LeViT_256', num_classes= 4, pretrained=False).to(device)
    
    for fold, (trn_idx, val_idx) in enumerate(IEC.folds):
        # we'll train fold 0 first
        if fold > 0:
          continue 

        print('Training with {} started'.format(fold))

        #print(len(trn_idx), len(val_idx))
        train_loader, val_loader = IEC.prepare_dataloader(train, trn_idx, val_idx, data_root='/content/iec-models/model_data/train_images')
        #print(len(train_loader),len(val_loader))

        #scaler = GradScaler()   
        optimizer = torch.optim.Adam(model.parameters(), lr=IEC.CFG['lr'], weight_decay=IEC.CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=IEC.CFG['T_0'], T_mult=1, eta_min=IEC.CFG['min_lr'], last_epoch=-1)
        
        loss_tr = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        
        for epoch in range(IEC.CFG['epochs']):
            IEC.train_one_epoch(fold, epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)
            model.eval()
            with torch.no_grad():
                IEC.valid_one_epoch(1, fold, epoch, model, loss_fn, train_loader, device, scheduler=None, schd_loss_update=False)
                IEC.valid_one_epoch(0, fold, epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
               
            #torch.save(model.state_dict(),'./content/result/originalViT_base16/{}_fold_{}_{}'.format(IEC.CFG['model_arch'], fold, epoch))      
            print("saving model...")