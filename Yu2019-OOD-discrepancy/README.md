# **Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy**

### **Experiment Order**
1. data_folder.py | original data to folderization
2. model/model_parm.py (option) | change old_version pretrained weight to new pretrained weight
3. train.py | train supervised/unsupervised learning
4. test.py | test supervised/unsupervised learning

### **data_folder.py**

1. Prerequsite
  : supervised dataset are saved 
      like {classname}_{ID}.{data_fomat} ex> 1_0001.jpg

2. Folderization 

    2-1.supdata
```sh
python data_folder.py --sup True 
--sup_path './dataset/cifar10' 
--num_unsup_tr 18000 
--num_unsup_val 1000 --num_unsup_val 1000
```

  - result

```sh
dataset/train/sup/class/*
       /val/sup/class/*
       /test/sup/class/*
``` 
   2-2. unsupdata
  
```sh
python data_folder.py 
--unsup True 
--unsup_path './dataset/LSUN/test'
--num_unsup_tr 9000 
--num_unsup_val 500 
--num_unsup_val 500
```
- result
```sh
dataset/train/unsup/*
       /val/unsup/*
       /test/unsup/*
```
### **model_parm.py**

1. Prerequsite

  : run Pytorch <=0.3.1  
  
2. Example

```sh
python model_parm.py 
--filein './ckp_weights/pre-train/weights/dense10.pth'
--fileout './ckp_weights/new-pre-train/dense10.pt'
--model densenet
```
- result

```sh
ckp_weights/new-pre-train/dense10.pt (can use in new Pytorch(>=1.0.0))
```

### **train.py**

1. Example

sup_train / not using pretrain model
```sh
python train.py 
--num_class 10 
--pre_train False 
--model densenet 
--sup_train True 
--sup_path './model/ckp_weights/new-pre-train/dense10.pt' 
```

sup_train / using pretrain model
```sh
python train.py 
--num_class 10 
--pre_train True 
--model_savedir ./model/ckp_weights/new-pre-train/dense10.pt
--model densenet --sup_train True 
```

unsup_train
```sh
python train.py 
--num_class 10 
--pre_train True 
--model_savedir ./model/ckp_weights/new-pre-train/dense10.pt
--model densenet 
--unsup_train True 
--unsup_path ./model/ckp_weights/fine-tune/weights/TINr_d_10.pt
```
2. Caution
- In supervised learning, we have to decide whether pretrained model is used
- When unsupervised learning, pretrain is True no matter supervised learning

### **test.py**

sup_test
```sh
python test.py 
--num_class 10 
--model densenet 
--sup_test True 
--sup_path './model/ckp_weights/new-pre-train/dense10.pt' 
```
  - result
```sh
'Test acc 94, 94'
```
unsup_test
```sh
python test.py 
--num_class 10 
--test test 
--model densenet 
--unsup_test True 
--unsup_path ./model/ckp_weights/fine-tune/weights/TINr_d_10.pt
```

  - result
```sh
        'UnSUP Test detection_err: 0.01
        'UnSUP Test AUROC: 99.0
        'UnSUP Test auprin: 99.0
        'UnSUP Test auprout: 99.0
        'UnSUP Test fpr95: 0.01
```

  - caution
  in unsup_test, we can use train data as test data or just test data as test data.
  (you can set this --test argument)

## **In-Distribution Datasets**
- CIFAR10
- CIFAR100

## **In-Distribution Datasets**
- TinyImageNet (crop, resize)
- LSUN (crop, resize)
- iSUN

## **Architecture**
- Dense-BC (depth = 100, growth rate = 12, dropout rate = 0)
- WideResNet (depth = 28, width = 10)

## **Metric**
- FPR at 95% TPR
- Detection Error
- AUROC
- AUPR (In, Out)

## **Parameters**
- margin : 1.2

## **Training Setup**

### **Supervised train**

**Architecture**|  **Dense_BC**  |  **WideResNet**
----------------|----------------|-----------------
epochs          |300             |200
batch_size      |64              |128
loss            |CrossEntropy    |CrossEntropy
optimizer(momentum)| SGD(momentum=0.9)|SGD(momentum=0.9)
learning rate|0.1|0.1
weight_decay|PaperX/0.00001|PaperX/0.00001
learning rate decay(epoch)|0.1(150,225)|0.1(100,150)

### **Unsupervised train**
**Architecture**|  **Dense_BC**  |  **WideResNet**
----------------|----------------|-----------------
epochs|10|10
batch_size|PaperX/64|PaperX/64
loss|Discrepancy Loss|Discrepancy Loss
optimizer(momentum)| PaperX/SGD(momentum=0.9)|PaperX/SGD(momentum=0.9)
learning rate|Paper 0.1/ Code 0.001|Paper 0.1/ Code 0.001
weight_decay|PaperX/0.00001|PaperX/0.00001
learning rate decay(epoch)|0.1(150,225)|0.1(100,150)
margin|1.2|1.2


## **Official Code or Reference Code**
- https://github.com/Mephisto405/Unsupervised-Out-of-Distribution-Detection-by-Maximum-Classifier-Discrepancy

## **Paper**
- https://arxiv.org/abs/1908.04951

## **Result**
- https://drive.google.com/file/d/1qEdPii-Lr4YOFbCdQjgt0rzNDqqkZYVA/view?usp=sharing
