# DMPHN-cvpr19-master
Pytorch Implementation of CVPR19 "[Deep Stacked Multi-patch Hierarchical Network for Image Deblurring](https://arxiv.org/pdf/1904.03468.pdf)" <br/>

Please download GoPro dataset into './datas'. <br/>
https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view

__Requires.__
```
pytorch-0.4.1
numpy
scipy
```

__For model training, run following commands.__

```
python xxx.py
```


__For model testing, copy test samples into './test_samples', then run following commands.__

```
python xxx_test.py
```
## Citation
If you use this code in your research, please cite the following paper. Thank you!

```
@InProceedings{Zhang_2019_CVPR,
author = {Zhang, Hongguang and Dai, Yuchao and Li, Hongdong and Koniusz, Piotr},
title = {Deep Stacked Hierarchical Multi-Patch Network for Image Deblurring},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
