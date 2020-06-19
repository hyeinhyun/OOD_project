import torch
import torch.nn as nn
from collections import OrderedDict
import warnings
warnings.filterwarnings(action='ignore') 
import argparse
import model.densenet as d
import model.wideresnet as wrn
import torch.nn.functional as F




class two_head_dense(nn.Module):
    def __init__(self,model,l_w1,l_w2):
        super().__init__()

        model=list(model.children())[:-1]
        self.net=nn.Sequential(*model)
        self.Linear1=l_w1
        self.Linear2=l_w2
    def forward(self,x):
        out=self.net(x)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1,342)
        out_1=self.Linear1(out)
        out_2=self.Linear2(out)
        
        return out_1,out_2
    
class two_head_wide(nn.Module):
    def __init__(self,model,l_w1,l_w2):
        super().__init__()

        model=list(model.children())[:-1]
        self.net=nn.Sequential(*model)
        self.Linear1=l_w1
        self.Linear2=l_w2
    def forward(self,x):
        out=self.net(x)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1,640)
        out_1=self.Linear1(out)
        out_2=self.Linear2(out)
        
        return out_1,out_2
    
    
def two_head_net(model,out_features,fileout='',pre_train=False):
    print(pre_train)
    if pre_train:
        if model=='densenet':
            two_head_net = d.DenseNet3(100,out_features).cuda()

            checkpoint = torch.load(fileout,map_location='cuda:0')
            two_head_net.load_state_dict(checkpoint)

            Linear=list(two_head_net.children())[-1]
            Linear=Linear.state_dict()
            Linear1=nn.Linear(in_features=342, out_features=out_features, bias=True)
            Linear1.load_state_dict(Linear)
            Linear1.cuda()
            Linear2=nn.Linear(in_features=342, out_features=out_features, bias=True)
            Linear2.load_state_dict(Linear)
            Linear2.cuda()

            two_head_net=two_head_dense(two_head_net,Linear1,Linear2)
        elif model=='wideresnet':
            two_head_net=wrn.WideResNet(out_features).cuda()
            checkpoint = torch.load(fileout,map_location='cuda:0')
            two_head_net.load_state_dict(checkpoint)

            Linear=list(two_head_net.children())[-1]
            Linear=Linear.state_dict()
            Linear1=nn.Linear(in_features=640,out_features=out_features, bias=True)
            Linear1.load_state_dict(Linear)
            Linear1.cuda()
            Linear2=nn.Linear(in_features=640, out_features=out_features, bias=True)
            Linear2.load_state_dict(Linear)
            Linear2.cuda()

            two_head_net=two_head_wide(two_head_net,Linear1,Linear2)
    else:
        if model=='densenet':
            two_head_net = d.DenseNet3(100,out_features).cuda()
            Linear1=nn.Linear(in_features=342, out_features=out_features, bias=True)
            Linear1.cuda()
            Linear2=nn.Linear(in_features=342, out_features=out_features, bias=True)
            Linear2.load_state_dict(Linear1.state_dict())
            Linear2.cuda()
            two_head_net=two_head_dense(two_head_net,Linear1,Linear2)
        elif model=='wideresnet':
            two_head_net=wrn.WideResNet(out_features).cuda()
            Linear1=nn.Linear(in_features=342, out_features=out_features, bias=True)
            Linear1.cuda()
            Linear2=nn.Linear(in_features=342, out_features=out_features, bias=True)
            Linear2.load_state_dict(Linear1.state_dict())
            Linear2.cuda()
            two_head_net=two_head_wide(two_head_net,Linear1,Linear2)
        
    return two_head_net
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model load parser')
    parser.add_argument('--pre_train',type=bool,default=True,help='whether pretrained model exist')
    parser.add_argument('--fileout', type=str, default='',help='name of saving state dict extracted model')
    parser.add_argument('--model', type=str, default='densenet',help='select densenet or wideresnet')
    parser.add_argument('--out_features',type=int, default=10,help='model ouput features')
    args = parser.parse_args()


    #model_load(args.model,args.filein,args.fileout)
    two_head_net(args.model,args.out_features,args.fileout,args.pre_train)



