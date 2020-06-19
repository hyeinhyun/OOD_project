#
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
        with torch.no_grad():
            for i, (input, label) in enumerate(dataloaders['unsup_test']):
                inputs = input.cuda()
                label = label.cuda()

                out_1, out_2 = model(inputs)
                score_1 = nn.functional.softmax(out_1, dim=1)
                score_2 = nn.functional.softmax(out_2, dim=1)
                disc = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((label.shape[0], ))

                discs=torch.cat((discs,disc))
                labels=torch.cat((labels,label))

            #labels = 1 - labels

            labels = labels.cpu()
            discs = discs.cpu()

            rocs,fpr95, detect_error, auprin, auprout = evaluate(labels, discs, metric='test',save_to=save_pic)
            print('Test AUROC: {:.3f}'.format(rocs))
            print('Test detection_err: {:.3f}'.format(detect_error))
            print('Test auprin: {:.3f}'.format(auprin))
            print('Test auprout: {:.3f}'.format(auprout))
            print('Test fpr95: {:.3f}'.format(fpr95))


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='test parser')

        #dataset
        parser.add_argument('--api', type=bool, default=False,help='using api data or not')
        parser.add_argument('--crop', type=bool, default=False,help='whether unsup data need crop or not')
        parser.add_argument('--ID_api', type=str,default='',help='ID data if using api')
        parser.add_argument('--sup_test_path', type=str,default='./dataset/test/sup/',help='test data path for sup exp')
        parser.add_argument('--unsup_test_id_path', type=str,default='./dataset/train/unsup/id/',help='test OOD data path for unsup exp')
        parser.add_argument('--unsup_test_id_path', type=str,default='./dataset/val/unsup/id/',help='test OOD data path for unsup exp')
        
        #model load
        parser.add_argument('--model', type=str, default='densenet',help='select densenet or wideresnet')
        
        #test
        parser.add_argument('--sup_test', type=bool, default=False,help='whether test supmodel or not')
        parser.add_argument('--sup_path',type=str,default='./model/ckp_weights/new-pre-train/dense10.pt',help='path saveing sup train weight')
        parser.add_argument('--unsup_test', type=bool, default=False,help='whether test unsup model or not')
        parser.add_argument('--unsup_path',type=str,default='./model/ckp_weights/fine-tune/weights/TINr_d_10.pt',help='path saveing sup train weight')
        parser.add_argument('--save_pic',type=str,default='./result/ROC.png',help='path saveing ROC curve')
        args = parser.parse_args()
        
        
        
        #########prepare######
        with open('configuration.json', 'a') as f:
            json.dump({'test':args.__dict__}, f, indent=2)
        data_transform = T.Compose([
        T.CenterCrop(size=32), 
        #T.ToTensor(),T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        num_class=len(os.listdir(args.sup_test_path))


        
        
        
        #########data load########
        sup_test=label_data(args.sup_test_path,data_transform,args.crop)
        unsup_test=unlabel_data(args.unsup_test_id_path,args.unsup_test_ood_path,data_transform,args.crop)
        dataset_loader=dataset_loader(sup_test=sup_test,unsup_test=unsup_test,BATCH=args.batch)
        
        #########model load#######
        model=two_head_net(args.model,num_class)
        
        ######### sup test########'./model/ckp_weights', 'fine-tune', 'weights'
        checkpoint = torch.load(args.sup_path,map_location='cuda:0')
        model.load_state_dict(checkpoint)
        acc_1,acc_2=sup_test_exp(model.cuda(), dataset_loader)
        print('Test acc {}, {}'.format(acc_1, acc_2))
        
        ######## unsup test #########
        checkpoint = torch.load(args.unsup_path,map_location='cuda:0')
        model.load_state_dict(checkpoint)
        rocs,fpr95, detect_error, auprin, auprout=unsup_test_exp(model.cuda(), dataset_loader,args.save_pic)
        print('UnSUP Test detection_err: {:.3f}'.format(detect_error))
        print('UnSUP Test AUROC: {:.3f}\n'.format(rocs))
        print('UnSUP Test auprin: {:.3f}\n'.format(auprin))
        print('UnSUP Test auprout: {:.3f}\n'.format(auprout))
        print('UnSUP Test fpr95: {:.3f}\n'.format(fpr95))
        
        #name check
        

        
        