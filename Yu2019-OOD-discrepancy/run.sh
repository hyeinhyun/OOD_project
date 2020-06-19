#run
"""
###################
#densenet10
###################
#LSUN_r / densenet / CIFAR 10
python  main.py --OOD_path='./dataset/LSUN_resize/LSUN_resize' --fileout='densenet10' --model_name='LSUNr_d_10' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10

#LSUN_C/ densenet / CIFAR 10
python  main.py --OOD_path='./dataset/LSUN/test' --crop=True --data_format='png' --fileout='densenet10' --model_name='LSUNc_d_10' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10

#TIN_r/ densenet / CIFAR 10
python  main.py --OOD_path='./dataset/Imagenet_resize/Imagenet_resize' --fileout='densenet10' --model_name='TINr_d_10' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10

#TIN_C/ densenet / CIFAR 10
python  main.py --OOD_path='./dataset/Imagenet/test' --crop=True --data_format='jpg' --fileout='densenet10' --model_name='TINc_d_10' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10
"""
#iSUN / densenet /CIFAR10
python  main.py --OOD_path='./dataset/iSUN/iSUN_patches' --data_format 'jpeg'  --fileout='densenet10' --model_name='iSUN_d_10' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10 --num_data=8925



"""
###################
#wideresnet10
###################

#LSUN_r / wideresnet / CIFAR 10
python  main.py --OOD_path='./dataset/LSUN_resize/LSUN_resize' --fileout='wideresnet10' --model_name='LSUNr_w_10' --model='wideresnet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10

#LSUN_C/ wideresnet / CIFAR 10
python  main.py --OOD_path='./dataset/LSUN/test' --crop=True --data_format='png' --fileout='wideresnet10' --model_name='LSUNc_w_10' --model='wideresnet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10

#TIN_r/ wideresnet / CIFAR 10
python  main.py --OOD_path='./dataset/Imagenet_resize/Imagenet_resize' --fileout='wideresnet10' --model_name='TINr_w_10' --model='wideresnet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10

#TIN_C/ wideresnet / CIFAR 10
python  main.py --OOD_path='./dataset/Imagenet/test' --crop=True --data_format='jpg' --fileout='wideresnet10' --model_name='TINc_w_10' --model='wideresnet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10
"""
#iSUN / densenet /CIFAR10
python  main.py --OOD_path='./dataset/iSUN/iSUN_patches' --data_format 'jpeg'  --fileout='wideresnet10' --model_name='iSUN_w_10' --model='wideresnet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=10 --num_data=8925

##################
#dense100
##################

#LSUN_r / densenet / CIFAR 100
python  main.py --ID_path='./dataset/cifar100' --OOD_path='./dataset/LSUN_resize/LSUN_resize' --fileout='densenet100' --model_name='LSUNr_d_100' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=100

#LSUN_C/ densenet / CIFAR 100
python  main.py --ID_path='./dataset/cifar100' --OOD_path='./dataset/LSUN/test' --crop=True --data_format='png' --fileout='densenet100' --model_name='LSUNc_d_100' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=100

#TIN_r/ densenet / CIFAR 100
python  main.py --ID_path='./dataset/cifar100' --OOD_path='./dataset/Imagenet_resize/Imagenet_resize' --fileout='densenet100' --model_name='TINr_d_100' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=100

#TIN_C/ densenet / CIFAR 100
python  main.py --ID_path='./dataset/cifar100' --OOD_path='./dataset/Imagenet/test' --crop=True --data_format='jpg' --fileout='densenet100' --model_name='TINc_d_100' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=100

#iSUN / densenet /CIFAR100
python  main.py --ID_path='./dataset/cifar100' --OOD_path='./dataset/iSUN/iSUN_patches' --data_format 'jpeg'  --fileout='densenet100' --model_name='iSUN_d_100' --model='densenet' --sup_test=True --unsup_train=True --unsup_test=True --out_features=100 --num_data=8925

