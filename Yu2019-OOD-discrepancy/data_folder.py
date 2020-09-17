import pickle
import os
import random
import shutil
import argparse
import json




def copy_files(file_list,from_dir,dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    #print(from_dir)
    #print(dst_dir)
    #distinct
    class_n=from_dir.split('/')[-2]
    for f_name in file_list:
        shutil.copy(from_dir+f_name,dst_dir+class_n+'_'+f_name)

        


def sup_folder(sup_path,dst_tr_path,dst_val_path,dst_test_path,dst_tr_path_un,dst_val_path_un,dst_test_path_un,ratio_sup_tr=0.6,ratio_sup_val=0.2,ratio_sup_test=0.2,ratio_unsup_tr=0.7,ratio_unsup_val=0.3,ratio_unsup_test=0):
    #{클래스명}_{고유아이디}.{data_fomat}형태로 저장되어있다고 가정.
    #label_test/ unlabel_test 용 먼저 나눔

    original_path=sup_path
    sup_list=os.listdir(original_path)
    #print(sup_list)
    total_len=0    
    for c in sup_list:
        #print(sup_list)
        class_path=original_path+c+'/'
        file_list=os.listdir(class_path)
        random.shuffle(file_list)
        length=len(file_list)
        total_len+=length
        tr_list=file_list[:int(length*ratio_sup_tr)]
        val_list=file_list[int(length*ratio_sup_tr):int(length*ratio_sup_tr)+int(length*ratio_sup_val)]
        test_list=file_list[int(length*ratio_sup_tr)+int(length*ratio_sup_val):]

        #sup exp tr
        dst_tr_class_path=dst_tr_path+str(c)+'/'
        copy_files(tr_list,class_path,dst_tr_class_path)

        
        #sup exp val
        dst_val_class_path=dst_val_path+str(c)+'/'
        copy_files(val_list,class_path,dst_val_class_path)

        #sup exp test
        dst_test_class_path=dst_test_path+str(c)+'/'
        copy_files(test_list,class_path,dst_test_class_path)

        
        #unsup exp id tr/val/test
        test_len=len(test_list)
        unsup_tr_list=test_list[:int(test_len*ratio_unsup_tr)]
        unsup_val_list=test_list[int(test_len*ratio_unsup_tr):int(test_len*ratio_unsup_tr)+int(test_len*ratio_unsup_val)]
        unsup_test_list=test_list[int(test_len*ratio_unsup_tr)+int(test_len*ratio_unsup_val):]
        copy_files(unsup_tr_list,class_path,dst_tr_path_un)
        copy_files(unsup_val_list,class_path,dst_val_path_un)
        copy_files(unsup_test_list,class_path,dst_test_path_un)

        print("SUP : class {} / total {} / train {} / val {} / test {} ".format(c,length,len(tr_list),len(val_list),len(test_list)))
        print("UNSUP : class {} / total {} / train {} / val {} / test {} ".format(c,test_len,len(unsup_tr_list),len(unsup_val_list),len(unsup_test_list)))
    print("Finish total {} data is moved".format(total_len))
        
    
    
def  unsup_folder(unsup_path,dst_tr_path,dst_val_path,dst_test_path,ratio_unsup_tr=0.7,ratio_unsup_val=0.3,ratio_unsup_test=0,classes=False):
    original_path=unsup_path

    unsup_list=os.listdir(original_path)


    if classes:
        total_len=0    
        for c in unsup_list:
            class_path=original_path+c+'/'
            file_list=os.listdir(class_path)
            random.shuffle(file_list)
            length=len(file_list)
            total_len+=length
            tr_list=file_list[:int(length*ratio_unsup_tr)]
            val_list=file_list[int(length*ratio_unsup_tr):int(length*ratio_unsup_tr)+int(length*ratio_unsup_val)]
            test_list=file_list[int(length*ratio_unsup_tr)+int(length*ratio_unsup_val):]
            

            copy_files(tr_list,class_path,dst_tr_path)
            copy_files(val_list,class_path,dst_val_path)
            copy_files(test_list,class_path,dst_test_path)

            print("class {} / total {} / train {} / val {} / test {} ".format(c,length,len(tr_list),len(val_list),len(test_list)))
        print("Finish total {} data is moved".format(total_len))


    #no classes 
    else:
        random.shuffle(unsup_list)        
        length=len(unsup_list)
        tr_list=unsup_list[:int(length*ratio_unsup_tr)]
        val_list=unsup_list[int(length*ratio_unsup_tr):int(length*ratio_unsup_tr)+int(length*ratio_unsup_val)]
        test_list=unsup_list[int(length*ratio_unsup_tr)+int(length*ratio_unsup_val):]
        
        copy_files(tr_list,original_path,dst_tr_path)
        copy_files(val_list,original_path,dst_val_path)
        copy_files(test_list,original_path,dst_test_path)


        
        print("Finish total {} / train {} / val {} / test {} ".format(length,len(tr_list),len(val_list),len(test_list)))


        
        
#unlabel_folder('./dataset/LSUN/test',9000,500,500)
#label_folder('./temp',1,1,1)


if __name__ == '__main__':
        #dataload
    parser = argparse.ArgumentParser(description='folder parser')
    parser.add_argument('--sup',type=bool,default=False, help='whether create sup folder')
    parser.add_argument('--sup_path', type=str, default='./dataset/cifar10/',help='path of sup data')
    parser.add_argument('--ratio_sup_tr', type=float, default=0.6,help='num of sup train data')
    parser.add_argument('--ratio_sup_val', type=float, default=0.2,help='num of sup val data')
    parser.add_argument('--ratio_sup_test', type=float, default=0.2,help='num of sup test data')
    
    parser.add_argument('--dst_sup_train', type=str, default='./dataset_exp4/train/sup/',help='dst sup exp train data')
    parser.add_argument('--dst_sup_val', type=str, default='./dataset_exp4/val/sup/',help='dst sup exp val data')
    parser.add_argument('--dst_sup_test', type=str, default='./dataset_exp4/test/sup/',help='dst sup exp test data')


    parser.add_argument('--unsup',type=bool,default=False, help='whether create sup folder')
    parser.add_argument('--classes', type=bool, default=False,help='whether ood data has classes or not')
    parser.add_argument('--unsup_path', type=str, default='./dataset/LSUN/test/',help='type of unsup data')
    
    parser.add_argument('--ratio_unsup_tr', type=float, default=0.9,help='num of unsup train data')
    parser.add_argument('--ratio_unsup_val', type=float, default=0.1,help='num of unsup val data')
    parser.add_argument('--ratio_unsup_test', type=float, default=0.0,help='num of unsup test data')
    
    parser.add_argument('--dst_unsup_train_id', type=str, default='./dataset_exp4/train/unsup/id/',help='dst unsup exp train id data')
    parser.add_argument('--dst_unsup_val_id', type=str, default='./dataset_exp4/val/unsup/id/',help='dst unsup exp val id data')
    parser.add_argument('--dst_unsup_test_id', type=str, default='./dataset_exp4/test/unsup/id/',help='dst unsup exp test id data')
    parser.add_argument('--dst_unsup_train_ood', type=str, default='./dataset_exp4/train/unsup/ood/',help='dst unsup exp train ood data')
    parser.add_argument('--dst_unsup_val_ood', type=str, default='./dataset_exp4/val/unsup/ood/',help='dst unsup exp val ood data')
    parser.add_argument('--dst_unsup_test_ood', type=str, default='./dataset_exp4/test/unsup/ood/',help='dst unsup exp test ood data')    
    
    args = parser.parse_args()
    with open('configuration.json', 'w') as f:
        json.dump({'data_folder':args.__dict__}, f, indent=2)
    
    if args.sup:
        sup_folder(args.sup_path,args.dst_sup_train,args.dst_sup_val,args.dst_sup_test,
                   args.dst_unsup_train_id,args.dst_unsup_val_id,args.dst_unsup_test_id,
                   args.ratio_sup_tr,args.ratio_sup_val,args.ratio_sup_test,args.ratio_unsup_tr,args.ratio_unsup_val,args.ratio_unsup_test)
        print("==========folder creating finish :)====================")
    if args.unsup:
        unsup_folder(args.unsup_path,args.dst_unsup_train_ood,args.dst_unsup_val_ood,args.dst_unsup_test_ood,args.ratio_unsup_tr,args.ratio_unsup_val,args.ratio_unsup_test,args.classes)
        print("==========folder creating finish :)====================")   
    

    
