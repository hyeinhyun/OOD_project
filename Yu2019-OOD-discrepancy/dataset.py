from PIL import Image
import os
import random
import os.path
import numpy as np
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data

#dataloader
import warnings
warnings.filterwarnings("ignore")

# Python
import random
import argparse
import json

# Torch
from torch.utils.data import DataLoader

# Torchvison
from torchvision.utils import make_grid
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100


OOD=1.
ID=0.

def api_sup_data(ID):
    if ID=='cifar10':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        sup_train = CIFAR10(ID, train=True, 
                            download=False, transform=train_transform)
        sup_val   = CIFAR10(ID, train=False, 
                                download=False, transform=test_transform)
        sup_test  = CIFAR10(ID, train=False, 
                                download=False, transform=test_transform)
    elif ID=='cifar100':
        train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        sup_train = CIFAR100(ID, train=True, 
                            download=False, transform=train_transform)
        sup_val   = CIFAR100(ID, train=False, 
                                download=False, transform=test_transform)
        sup_test  = CIFAR100(ID, train=False, 
                                download=False, transform=test_transform)
    return sup_train,sup_val,sup_test



class label_data(data.Dataset):
    def __init__(self,sup_path,d_size,transform=None):
        #self.ood = OOD_path
        #self.id = ID_path
        self.transform = transform
        self.data = []
        self.targets = []
        self.to_tensor=T.Compose([T.ToTensor()])
        self.sup_path=sup_path
        num_class=len(os.listdir(self.sup_path))
        
        try:
            for c in range(num_class):
                class_path=self.sup_path+'{}/'.format(str(c))
                #print(class_path)
                for name in (os.listdir(class_path)):
                   #print(class_path+name)
                    img=Image.open(class_path+name).convert(mode='RGB')
                    img =self.transform(img)
                    np_img = np.array(img)

                    self.data.append(np_img)
                    self.targets.append(c)
            print(np.vstack(self.data).reshape(-1, d_size, d_size, 3).shape)
            self.data = np.vstack(self.data).reshape(-1, d_size, d_size, 3)

        except:
            raise NotImplementedError('')          
            
        self.targets = np.array(self.targets)

        # shuffle
        indices = list(range(len(self.targets)))
        random.Random(4).shuffle(indices)
        self.data = self.data[indices]
        self.targets = self.targets[indices]
                          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            img = Image.fromarray(self.data[index])
            img = Image.fromarray(self.data[index])
        except:
            img = Image.fromarray((self.data[index] * 255).astype(np.uint8))

        if self.transform is not None:
            img = self.to_tensor(img)

        return img, self.targets[index]
    
    
    
class unlabel_data(data.Dataset):
    def __init__(self,id_path,ood_path,d_size,transform=None):
        #self.ood = OOD_path
        #self.id = ID_path
        self.transform = transform
        self.data = []
        self.targets = []
        self.classes=[]
        self.to_tensor=T.Compose([T.ToTensor()])
        self.id_path=id_path
        num_class=len(os.listdir(self.id_path))
        self.ood_path=ood_path

        try:
            #sup data
            #for c in range(1,num_class+1):
                #class_path=id_path+'{}/'.format(str(c))
            for name in (os.listdir(self.id_path)):
                cls=name.split('_')[0]
                img=Image.open(self.id_path+name).convert(mode='RGB')
                img = transform(img)
                np_img = np.array(img)
                self.classes.append(cls)
                self.data.append(np_img)
                self.targets.append(ID)
                #self.data = np.vstack(self.data).reshape(-1, 32, 32, 3)
            
            #unsup data
            for name in (os.listdir(self.ood_path)):
                cls=name.split('_')[0]
                img=Image.open(self.ood_path+name).convert(mode='RGB')
                img=transform(img)
                np_img = np.array(img)
                self.classes.append(cls)
                self.data.append(np_img)
                self.targets.append(OOD)
            self.data = np.vstack(self.data).reshape(-1, d_size, d_size, 3)
            print(len(self.targets))

        except:
            raise NotImplementedError('')          

        self.targets = np.array(self.targets)
        self.classes = np.array(self.classes)

        # shuffle
        indices = list(range(len(self.targets)))
        random.Random(4).shuffle(indices)
        self.data = self.data[indices]
        self.targets = self.targets[indices]
        self.classes=self.classes[indices]
                          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            img = Image.fromarray(self.data[index])
        except:
            img = Image.fromarray((self.data[index] * 255).astype(np.uint8))

        if self.transform is not None:
            img = self.to_tensor(img)

        return img, self.targets[index],self.classes[index]
    
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataload parser')
    parser.add_argument('--api', type=bool, default=False,help='using api data or not')
    parser.add_argument('--ID_api', type=str,default='',help='ID data if using api')
   #parser.add_argument('--crop', type=bool, default=False,help='whether unsup data need crop or not')
    parser.add_argument('--sup_path',type=str,default='./dataset/train/sup/',help='dataset path for sup experiment')
    parser.add_argument('--id_path', type=str,default='./dataset/train/unsup/id/',help='id path for unsup experiment')
    parser.add_argument('--ood_path', type=str,default='./dataset/train/unsup/ood/',help='ood path for unsup experiment')
    parser.add_argument('--d_size',type=int,default=32,help='data size')

    args = parser.parse_args()
    with open('configuration.json', 'a') as f:
        json.dump({'dataset':args.__dict__}, f, indent=2)
    data_transform = T.Compose([
    T.CenterCrop(size=args.d_size), 
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    if args.api:
        api_sup_data(args.ID_api)
    else:
        sup_data=label_data(args.sup_path,args.d_size,transform=None)
        #unsup_train=unlabel_data('train',args.num_class,data_transform,args.crop)
        #unsup_val=unlabel_data('val',args.num_class,data_transform,args.crop)
        unsup_data=unlabel_data(args.id_path,args.ood_path,args.d_size,data_transform) 
