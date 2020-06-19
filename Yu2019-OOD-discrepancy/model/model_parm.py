#should run old version torch 0.3.1
import torch
import torch.nn as nn
from collections import OrderedDict
import warnings
warnings.filterwarnings(action='ignore') 
import argparse
import densenet as d
import wideresnet as wrn




def model_load(model,filein,fileout):
    if model=='densenet':
        pre_train=filein
        two_head_net=torch.load(pre_train,map_location='cuda:0')
        two_head_net=two_head_net.state_dict()
        new_pre_train=fileout
        torch.save(two_head_net,new_pre_train)
        print("***************Weight is saved in {} *******************".format(new_pre_train))
        return 0
    elif model =="wideresnet":
        pre_train=filein
        two_head_net=torch.load(pre_train,map_location='cuda:0')    
        two_head_net=two_head_net.state_dict()
        new_state_dict = OrderedDict()
        for k, v in two_head_net.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        new_pre_train=fileout
        torch.save(new_state_dict,new_pre_train)
        print("***************Weight is saved in {} *******************".format(new_pre_train))
        return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model load parser')
    parser.add_argument('--filein', type=str, default='./ckp_weights/pre-train/weights/dense10.pth',help='name of pretrained model')
    parser.add_argument('--fileout', type=str, default='./ckp_weights/new-pre-train/dense10.pt',help='name of saving state dict extracted model')
    parser.add_argument('--model', type=str, default='densenet',help='select densenet or wideresnet')
    args = parser.parse_args()


    model_load(args.model,args.filein,args.fileout)
    #two_head_net(args.model,args.fileout)