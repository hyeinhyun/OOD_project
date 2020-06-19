#
import os
import random
import argparse

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Utils
import visdom
from tqdm import tqdm
import json
#Custom
from dataset import *
from dataloader import *
from model.model_match import two_head_net


iters = 0

#
def DiscrepancyLoss(input_1, input_2, m = 1.2):
    soft_1=nn.functional.softmax(input_1, dim=1)
    soft_2=nn.functional.softmax(input_2, dim=1)
    entropy_1=-soft_1*nn.functional.log_softmax(input_1, dim=1)
    entropy_2=-soft_2*nn.functional.log_softmax(input_2, dim=1)
    entropy_1=torch.sum(entropy_1, dim=1)
    entropy_2=torch.sum(entropy_2, dim=1)

    loss = torch.nn.ReLU()(m - torch.mean(entropy_1 - entropy_2))
    return loss


def train_epoch(model, criterions, optimizer, scheduler, dataloaders, num_epochs):
    model.train()

    global iters

    for data in tqdm(dataloaders['sup_train'], leave=False, total=len(dataloaders['sup_train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizer.zero_grad()

        out_1, out_2 = model(inputs)
        loss = criterions['sup'](out_1, labels) + criterions['sup'](out_2, labels)

        loss.backward()

def sup_train_exp(model, criterions, optimizer, scheduler, dataloaders, num_epochs,sup_path):
    """
    Supervised training step.
    """
    print('>> Train a Model.')
    best_acc = 0.

    for epoch in range(num_epochs):
        scheduler.step()

        train_epoch(model, criterions, optimizer, scheduler, dataloaders, num_epochs)
        
        #validation
        model.eval()
        total = 0
        correct_1 = 0
        correct_2 = 0

        with torch.no_grad():
            for (inputs, labels) in dataloaders['sup_val']:
                inputs = inputs.cuda()
                labels = labels.cuda()

                out_1, out_2 = model(inputs)
                _, pred_1 = torch.max(out_1.data, 1)
                _, pred_2 = torch.max(out_2.data, 1)
                total += labels.size(0)
                correct_1 += (pred_1 == labels).sum().item()
                correct_2 += (pred_2 == labels).sum().item()

            # Save a checkpoint
        acc_1, acc_2 = (100 * correct_1 / total), (100 * correct_2 / total)
        if best_acc < acc_1:
            best_acc = acc_1
            torch.save({model.state_dict()},'{}'.format(sup_path))
        print('Val Accs: {:.3f}, {:.3f} \t Best Acc: {:.3f}'.format(acc_1, acc_2, best_acc))
    print('>> Finished.')

def unsup_train_exp(model, criterions, optimizer, scheduler, dataloaders,unsup_path,num_epochs=10, vis=None):
    """
    Unsupervised fine-tuning step with supervised guidance.
    """
    print('>> Fine-tune a Model.')
    best_roc = 0.0
    model_name = model_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    iters = 0
    plot_data = {'X': [], 'Y': [], 'legend': ['Sup. Loss', 'Unsup. Loss', 'Tot. Loss']}

    for epoch in range(num_epochs):
        scheduler.step()
        # Training
        model.train()
        for i, sup_data in enumerate(dataloaders['sup_train']):
            unsup_data = dataloaders['unsup_train'][i % len(dataloaders['unsup_train'])]
            sup_inputs = sup_data[0]
            sup_labels = sup_data[1].cuda()
            unsup_inputs = unsup_data[0]
            data_inputs=torch.cat((sup_inputs,unsup_inputs),axis=0).cuda()
            
            # unsup_labels = unsup_data[1].cuda()
            iters += 1

            optimizer.zero_grad()
            out_1, out_2 = model(data_inputs)
            sup_out_1,sup_out_2=out_1[:sup_inputs.shape[0]],out_2[:sup_inputs.shape[0]]
            unsup_out_1,unsup_out_2=out_1[sup_inputs.shape[0]:],out_2[sup_inputs.shape[0]:]

            loss_sup = criterions['sup'](sup_out_1, sup_labels) + criterions['sup'](sup_out_2, sup_labels) # Step A

            loss_unsup = criterions['unsup'](unsup_out_1, unsup_out_2) # Step B
            loss = loss_unsup + loss_sup
            loss.backward()
            optimizer.step()

            # visualize
            if (iters % 10 == 0) and (vis != None) and (plot_data != None):
                plot_data['X'].append(iters)
                plot_data['Y'].append([
                    loss_sup.item(),
                    loss_unsup.item(),
                    loss.item()
                ])
                vis.line(
                    X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                    Y=np.array(plot_data['Y']),
                    opts={
                        'title': 'Loss over Time',
                        'legend': plot_data['legend'],
                        'xlabel': 'Iterations',
                        'ylabel': 'Loss',
                        'width': 1200,
                        'height': 390,
                    },
                    win=2
                )
        # Validate
    model.eval()
    labels = torch.zeros((0, )).cuda() # initialize
    discs = torch.zeros((0, )).cuda() 
    with torch.no_grad():
        for i, (input, label) in enumerate(dataloaders['unsup_test']):
            inputs = input.cuda()
            label = label.cuda()

            out_1, out_2 = model(inputs)
            score_1 = nn.functional.softmax(out_1, dim=1)
            score_2 = nn.functional.softmax(out_2, dim=1)
            disc = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((label.shape[0], ))
            
            discs=torch.cat((discs,disc))
            label=label.reshape((label.shape[0], ))
            labels=torch.cat((labels,label))
    
        #labels = 1 - labels

        labels = labels.cpu()
        discs = discs.cpu()
        
        roc = evaluate(labels.cpu(), discs.cpu(), metric='roc')
        print('Epoch{} AUROC: {:.3f}'.format(epoch, roc))
        if best_roc < roc:
            best_roc = roc
            torch.save({ model.state_dict()},'{}'.format(unsup_path))
            print('Model saved.')
    print('>> Finished.')

    
    
if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='train parser')
        vis = visdom.Visdom(server='http://localhost')
        #dataset
        parser.add_argument('--api', type=bool, default=False,help='using api data or not')
        parser.add_argument('--crop', type=bool, default=False,help='whether unsup data need crop or not')
        parser.add_argument('--ID_api', type=str,default='',help='ID data if using api')
        parser.add_argument('--sup_tr_path', type=str,default='./dataset/train/sup/',help='train data path for sup exp')
        parser.add_argument('--sup_val_path', type=str,default='./dataset/val/sup/',help='val data path for sup exp')
        parser.add_argument('--unsup_tr_id_path', type=str,default='./dataset/train/unsup/id/',help='train ID data path for unsup exp')
        parser.add_argument('--unsup_val_id_path', type=str,default='./dataset/val/unsup/id/',help='val ID data path for unsup exp')
        parser.add_argument('--unsup_tr_ood_path', type=str,default='./dataset/train/unsup/ood/',help='train OOD data path for unsup exp')
        parser.add_argument('--unsup_val_ood_path', type=str,default='./dataset/val/unsup/ood/',help='val OOD data path for unsup exp')

        
        
        #model load
        parser.add_argument('--pre_train',type=bool,default=True,help='whether pretrained model exist')
        parser.add_argument('--p_weight_path', type=str, default='',help='pretrained weight path')
        parser.add_argument('--model', type=str, default='densenet',help='select densenet or wideresnet')

        
        #sup_train
        parser.add_argument('--sup_train', type=bool, default=False,help='whether do supervised train or not')
        parser.add_argument('--sup_path', type=str, default='./model/ckp_weights/new-pre-train/dense10.pt',help='path of saving state dict sup_train model')
        parser.add_argument('--sup_epoch', type=int, default=10,help='supervised train epoch')
        parser.add_argument('--sup_lr', type=float, default=0.01,help='supervised train learning rate')
        parser.add_argument('--sup_momentum', type=float, default=0.9,help='supervised train momentum')
        parser.add_argument('--sup_wdecay', type=float, default=5e-4,help='supervised train weight decay')        
        #unsup_train
        parser.add_argument('--unsup_train', type=bool, default=False,help='whether do unsupervised train or not')
        parser.add_argument('--unsup_path', type=str, default='./model/ckp_weights/fine-tune/weights/TINr_d_10.pt',help='path of saving state dict unsup_train model')
        parser.add_argument('--unsup_epoch', type=int, default=10,help='unsupervised train epoch')
        parser.add_argument('--unsup_lr', type=float, default=0.01,help='unsupervised train learning rate')
        parser.add_argument('--unsup_momentum', type=float, default=0.9,help='unsupervised train momentum')
        parser.add_argument('--unsup_wdecay', type=float, default=5e-4,help='unsupervised train weight decay')
        
        parser.add_argument('--batch',type=int, default=64,help='batch size')
        args = parser.parse_args()
        
        with open('configuration.json', 'a') as f:
            json.dump({'train':args.__dict__}, f, indent=2)
        
        num_class=len(os.listdir(args.sup_tr_path))
        data_transform = T.Compose([
            T.CenterCrop(size=32), 
            #T.ToTensor(),T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        sup_criterion = nn.CrossEntropyLoss()
        unsup_criterion = DiscrepancyLoss
        criterions = {'sup': sup_criterion, 'unsup': unsup_criterion}
        MILESTONES = [150, 225]

        #########data load########
        sup_train=label_data(args.sup_tr_path,data_transform,args.crop)
        sup_val=label_data(args.sup_val_path,data_transform,args.crop)
        unsup_train=unlabel_data(args.unsup_tr_id_path,args.unsup_tr_ood_path,data_transform,args.crop)
        unsup_val=unlabel_data(args.unsup_val_id_path,args.unsup_val_ood_path,data_transform,args.crop)         
        dataset_loader=dataset_loader(sup_train=sup_train,sup_val=sup_val,unsup_train=unsup_train,unsup_val=unsup_val,BATCH=args.batch)
        
        ##########model load #########

        model=two_head_net(args.model,num_class,args.p_weight_path,args.pre_train)
            
        ############sup train##########
        if args.sup_train:
            if args.pre_train:
                print("using pre trained model, do not need to train again!")
                pass
            else:
                optimizer = optim.SGD(model.parameters(), lr=args.sup_lr, 
                              momentum=args.sup_momentum, weight_decay=args.sup_wdecay)
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
                sup_train_exp(model.cuda(),criterions,optimizer,scheduler,dataset_loader,args.sup_epoch,args.sup_path)
                print("========sup train fininsh!=============")
                
        ############unsup train###########
        if args.unsup_train:
            optimizer = optim.SGD(two_head_net.parameters(), 
                  lr=args.unsup_lr,
                  momentum=args.unsup_momentum, weight_decay=args.unsup_wdecay)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
            unsup_train_exp(model.cuda(), criterions, optimizer, scheduler, dataset_loader,unsup_path=args.unsup_path, num_epochs=args.unsup_epoch, vis=vis)
            
