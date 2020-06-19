## model_parm.py
this module extract weights that created in old pytorch to use in pytorch version <=1.4.0

## Download pre trained model (from ODIN)

**put those into model/ckp_weights/pre-train/weights/**

[DenseNet-BC trained on CIFAR-10](https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz)

[DenseNet-BC trained on CIFAR-100](https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz)

[Wide ResNet trained on CIFAR-10](https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz)

Wide ResNet trained on CIFAR-100 (not exist)

## 1. First run model_parm.py

this file get weight from above files, and save weights into model/ckp_weight/new_pre-train/

**should run in pytorch below 0.3.1**

