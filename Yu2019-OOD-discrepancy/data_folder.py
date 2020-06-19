import pickle
import os
import random
import shutil
import argparse
import json



def sup_folder(sup_path,num_sup_tr,num_sup_val,num_sup_test):
    #{클래스명}_{고유아이디}.{data_fomat}형태로 저장되어있다고 가정.
    
    original_path=sup_path+'/'
    sup_list=os.listdir(original_path)
    random.shuffle(sup_list)
    
    
    for f_name in sup_list[:num_sup_tr]:
        c_name=f_name.split('_')[0]
        mv_path='./dataset/train/sup/{}/'.format(c_name)
        if not os.path.exists(mv_path):
            os.makedirs(mv_path)
        shutil.copy(original_path+f_name, mv_path+f_name)
        
    for f_name in sup_list[num_sup_tr:num_sup_tr+num_sup_val]:
        c_name=f_name.split('_')[0]
        mv_path='./dataset/val/sup/{}/'.format(c_name)
        if not os.path.exists(mv_path):
            os.makedirs(mv_path)
        shutil.copy(original_path+f_name, mv_path+f_name)
        
    for f_name in sup_list[num_sup_tr+num_sup_val:]:
        c_name=f_name.split('_')[0]
        mv_path='./dataset/test/sup/{}/'.format(c_name)
        if not os.path.exists(mv_path):
            os.makedirs(mv_path)
        shutil.copy(original_path+f_name, mv_path+f_name)



def  unsup_folder(unsup_path,num_unsup_tr,num_unsup_val,num_unsup_test):
    #pre-processing
    original_path=unsup_path+'/'
    unsup_list=os.listdir(original_path)
    random.shuffle(unsup_list)
    
    
    for f_name in unsup_list[:num_unsup_tr]:
        #copy
        shutil.copy(original_path+f_name, './dataset/train/unsup/'+f_name)
    for i in unsup_list[num_unsup_tr:num_unsup_tr+num_unsup_val]:
        #copy
        shutil.copy(original_path+f_name, './dataset/val/unsup/'+f_name)
    for i in unsup_list[num_unsup_tr+num_unsup_val:]: 
        shutil.copy(original_path+f_name, './dataset/test/unsup/'+f_name)
        
#unlabel_folder('./dataset/LSUN/test',9000,500,500)
#label_folder('./temp',1,1,1)


if __name__ == '__main__':
        #dataload
    parser = argparse.ArgumentParser(description='folder parser')
    parser.add_argument('--sup',type=bool,default=False, help='whether create sup folder')
    parser.add_argument('--sup_path', type=str, default='./dataset/cifar10',help='path of sup data')
    parser.add_argument('--num_sup_tr', type=int, default=10000,help='num of sup train data')
    parser.add_argument('--num_sup_val', type=int, default=10000,help='num of sup val data')
    parser.add_argument('--num_sup_test', type=int, default=10000,help='num of sup test data')

    parser.add_argument('--unsup',type=bool,default=False, help='whether create sup folder')
    parser.add_argument('--unsup_path', type=str, default='./dataset/LSUN/test',help='type of unsup data')
    parser.add_argument('--num_unsup_tr', type=int, default=10000,help='num of unsup train data')
    parser.add_argument('--num_unsup_val', type=int, default=10000,help='num of unsup val data')
    parser.add_argument('--num_unsup_test', type=int, default=10000,help='num of unsup test data')
    
    args = parser.parse_args()
    with open('configuration.json', 'w') as f:
        json.dump({'data_folder':args.__dict__}, f, indent=2)
    
    if args.sup:
        sup_folder(args.sup_path,args.num_sup_tr,args.num_sup_val,args.num_sup_test)
        print("==========folder creating finish :)====================")
    if args.unsup:
        unsup_folder(args.unsup_path,args.num_unsup_tr,args.num_unsup_val,args.num_unsup_test)
        print("==========folder creating finish :)====================")   
    

    
