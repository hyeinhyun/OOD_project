 #
#
import os
import random
import argparse

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from torchvision.datasets import ImageFolder

# Utils
import visdom
from tqdm import tqdm
import json
#Custom
from dataset import *
from dataloader import *
from model.model_match import two_head_net
from metrics import *
from metric2 import *

def sup_test_exp(model, dataloaders):
    """
    Test the baseline two-headed network right after the the supervised pre-training step.

    Sanity check:
        The accuracy will be higher than 92.5 at least.
    """
    model.eval()

    total = 0
    correct_1 = 0
    correct_2 = 0

    with torch.no_grad():
        for (inputs, labels) in dataloaders['sup_test']:
            inputs = inputs.cuda()
            labels = labels.cuda()

            out_1, out_2 = model(inputs)
            _, pred_1 = torch.max(out_1.data, 1)
            _, pred_2 = torch.max(out_2.data, 1)
            total += labels.size(0)
            correct_1 += (pred_1 == labels).sum().item()
            correct_2 += (pred_2 == labels).sum().item()
    
    return (100 * correct_1 / total), (100 * correct_2 / total)


def unsup_test_exp(model, dataloaders,save_pic):
    
    model.eval()
    labels = torch.zeros((0, )).cuda() # initialize
    discs = torch.zeros((0, )).cuda() 
    pred = torch.zeros((0, )).cuda() 

    with torch.no_grad():
        for i, (input, label,cls) in enumerate(dataloaders['unsup_test']):
            inputs = input.cuda()
            label = label.cuda()

            out_1, out_2 = model(inputs)
            pred_1 = F.softmax(out_1, 1)
            pred_1,_=torch.max(pred_1,1)
            score_1 = F.softmax(out_1, dim=1)
            score_2 = F.softmax(out_2, dim=1)
            disc = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((label.shape[0], ))

            discs=torch.cat((discs,disc))
            label=label.reshape((label.shape[0], )).float()
            labels=torch.cat((labels,label))
            pred_1=pred_1.reshape((pred_1.shape[0], )).float()
            pred=torch.cat((pred,pred_1))
    labels = labels.cpu()
    discs = discs.cpu()
    """
    pred=pred.cpu() #for fit my metric
    X1=pred[labels==0].tolist()
    Y1=pred[labels==1].tolist()
    total=X1+Y1
    total.sort()
    diff=np.array(total)
    print("tpr %f" % (tpr95(X1,Y1,diff)))
    print("auprin %f" % auprIn(X1,Y1,diff))
    print("auroc %f" %auroc(X1,Y1,diff))
    print("auprout %f" %auprOut(X1,Y1,diff))
    print("detec %f" %detection(X1,Y1,diff))
    """
    rocs,fpr95, detect_error, auprin, auprout = evaluate(labels, discs, metric='test',save_to=save_pic)
    return rocs,fpr95, detect_error, auprin, auprout

def ab_test_exp(model, dataloaders,save_pic):
    thre= 0.6011339426040649

    model.eval()
    labels = torch.zeros((0, )).cuda() # initialize
    discs = torch.zeros((0, )).cuda() 
    pred = torch.zeros((0, )).cuda() 

    clss = [] 

    with torch.no_grad():
        for i, (input, label,cls) in enumerate(dataloaders['unsup_test']):
            inputs = input.cuda()
            label = label.cuda()

            out_1, out_2 = model(inputs)
            pred_1 = F.softmax(out_1, 1)
            pred_1,_=torch.max(pred_1,1)
            score_1 = F.softmax(out_1, dim=1)
            score_2 = F.softmax(out_2, dim=1)
            disc = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((label.shape[0], ))

            discs=torch.cat((discs,disc))
            label=label.reshape((label.shape[0], )).float()
            labels=torch.cat((labels,label))
            pred_1=pred_1.reshape((pred_1.shape[0], )).float()
            pred=torch.cat((pred,pred_1))
            clss+=cls
    #labels = 1 - labels

    labels = labels.cpu()
    discs = discs.cpu()
    pred=pred.cpu()
    
    for c in list(set(clss)):
        print(labels)
        r_label=labels[np.array(clss)==c]
        r_dist=pred[np.array(clss)==c]
        X1=r_dist[r_label==0]#in
        Y1=r_dist[r_label==1]#out

        tp = sum(np.array(X1) >= thre)
        fn = sum(np.array(X1) < thre)
        tn = sum(np.array(Y1) < thre)
        fp = sum(np.array(Y1) >= thre)
        print(c,' : ')
        print(len(r_label))
        print('\ntp: %d, fn: %d , tn: %d, fp: %d\n' % (tp, fn, tn, fp))     
        print('\ncor: %d , incor: %d\n' % (tp+tn, fn+fp))        



    #print('Test AUROC: {:.3f}'.format(rocs))
    #print('Test detection_err: {:.3f}'.format(detect_error))
    #print('Test auprin: {:.3f}'.format(auprin))
    #print('Test auprout: {:.3f}'.format(auprout))
    #print('Test fpr95: {:.3f}'.format(fpr95))

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='test parser')

        #dataset
        parser.add_argument('--api', type=bool, default=False,help='using api data or not')
        #parser.add_argument('--crop', type=bool, default=False,help='whether unsup data need crop or not')
        parser.add_argument('--ID_api', type=str,default='',help='ID data if using api')
        parser.add_argument('--sup_test_path', type=str,default='./dataset/test/sup/',help='test data path for sup exp')
        parser.add_argument('--unsup_test_id_path', type=str,default='./dataset/test/unsup/id/',help='test OOD data path for unsup exp')
        parser.add_argument('--unsup_test_ood_path', type=str,default='./dataset/test/unsup/ood/',help='test OOD data path for unsup exp')
        parser.add_argument('--d_size',type=int,default=32,help='data size')        
        #model load
        parser.add_argument('--model', type=str, default='densenet',help='select densenet or wideresnet')
        
        #test
        parser.add_argument('--sup_test', type=bool, default=False,help='whether test supmodel or not')
        parser.add_argument('--sup_path',type=str,default='./model/ckp_weights/new-pre-train/dense10.pt',help='path saveing sup train weight')
        parser.add_argument('--unsup_test', type=bool, default=False,help='whether test unsup model or not')
        parser.add_argument('--unsup_path',type=str,default='./model/ckp_weights/fine-tune/weights/TINr_d_10.pt',help='path saveing sup train weight')
        parser.add_argument('--save_pic',type=str,default='./result/ROC.png',help='path saveing ROC curve')
        parser.add_argument('--batch',type=int,default=64,help='batch_size')
        args = parser.parse_args()
        
        
        
        #########prepare######
        with open('configuration.json', 'a') as f:
            json.dump({'test':args.__dict__}, f, indent=2)
        """
        data_transform = T.Compose([
        T.CenterCrop(size=32), 
        #T.ToTensor(),T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        """
        data_transform = T.Compose([
            T.Resize(args.d_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #T.ToTensor()
            #T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        num_class=len(os.listdir(args.sup_test_path))


        
        
        
        #########data load########
        sup_test=''
        unsup_test=''
        if args.sup_test:
            sup_test=ImageFolder(args.sup_test_path,transform=data_transform)
        if args.unsup_test:
            unsup_test=unlabel_data(args.unsup_test_id_path,args.unsup_test_ood_path,args.d_size,data_transform)
        dataset_loader=dataset_loader(sup_test=sup_test,unsup_test=unsup_test,BATCH=args.batch)
        
        #########model load#######
        model=two_head_net(args.model,num_class)
        
        ######### sup test########'./model/ckp_weights', 'fine-tune', 'weights'
        if args.sup_test:
            checkpoint = torch.load(args.sup_path,map_location='cuda:0')
            model.load_state_dict(checkpoint)
            acc_1,acc_2=sup_test_exp(model.cuda(), dataset_loader)
            print('Test acc {}, {}'.format(acc_1, acc_2))
        
        ######## unsup test #########
        if args.unsup_test:
            checkpoint = torch.load(args.unsup_path,map_location='cuda:0')
            model.load_state_dict(checkpoint)
            ab_test_exp(model.cuda(), dataset_loader,args.save_pic)

            rocs,fpr95, detect_error, auprin, auprout=unsup_test_exp(model.cuda(), dataset_loader,args.save_pic)
            print('UnSUP Test detection_err: {:.3f}'.format(detect_error))
            print('UnSUP Test AUROC: {:.3f}\n'.format(rocs))
            print('UnSUP Test auprin: {:.3f}\n'.format(auprin))
            print('UnSUP Test auprout: {:.3f}\n'.format(auprout))
            print('UnSUP Test fpr95: {:.3f}\n'.format(fpr95))
        
        #name check
        

        
        

