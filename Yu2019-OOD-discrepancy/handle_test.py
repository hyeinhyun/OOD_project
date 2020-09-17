import os
import shutil
import random
from_dir='./dataset2/test/sup/'
train_dir='./dataset2/train/unsup/id/'
val_dir='./dataset2/val/unsup/id/'
file_li=[]
for c in os.listdir(from_dir):
    file_li+=os.listdir(from_dir+c)

print(len(file_li))
random.shuffle(file_li)

for i in file_li[:3000]:
    c=i.split('_')[0]
    shutil.copy(from_dir+c+'/'+i,train_dir+i)
for i in file_li[3000:]:
    c=i.split('_')[0]
    shutil.copy(from_dir+c+'/'+i,val_dir+i)
