# When do curricula work?
This repository provides the sample codes for the paper 
[When do curricula work](https://openreview.net/forum?id=tW4QEInpni)
which was accepted at ICLR 2021 for Oral Presentation. It contains the code for experiments on CIFAR10, CIFAR100 and CIFAR100 with noisy labels.

## Citation
If you find this code useful in your research then please cite:
```
@inproceedings{
wu2021when,
title={When Do Curricula Work?},
author={Xiaoxia Wu and Ethan Dyer and Behnam Neyshabur},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=tW4QEInpni}
}
```

## Baseline
 ```python main_w_test.py --ordering standard```
This command is our standard setup (ResNet50 architecture with SGD Momentum and cosine scheduler) on CIFAR10 dataset with @ 100 epoch.
 
## Curricula includes curriculum, anti-curriculum and random-curriculum learning
### curriculum: linear pacing function with paremeter a=0.2 and b=0.8
```python main_w_test.py --pacing-a 0.2 --pacing-b 0.8 --pacing-f linear --half```
### anti-curriculum: linear pacing function with paremeter a=0.2 and b=0.8
``` python main_w_test.py --pacing-a 0.2 --pacing-b 0.8 --pacing-f linear --ordering anti-curr --half```
### random-curriculum: linear pacing function with paremeter a=0.2 and b=0.8
``` python main_w_test.py --pacing-a 0.2 --pacing-b 0.8 --pacing-f linear --ordering random --half```

###### Note  "main_wo_test.py" contains no test set but validation set only, while "main_w_test.py" contains validation and test dataset. Our paper uses the best validation accuracy to pick the corresponding test performance (i.e., main_w_test.py).


### options:
| argument                    | options                                  |
|-----------------------------|------------------------------------------|
| --dataset                   | `cifar10(default)  cifar100 cifar100N`   |
| --rand-fraction             | `0(default) 0.2 0.4 0.6 0.8`             |
| --ordering                  | `curr(default) anti-curr random standard`|
| --pacing-f                  | `linear(default) quad root step exp log` |
| --pacing-a                  | `[0,10] 1(default)`                      |
| --pacing-b                  | `[0,1] 1(default)`                       |
| --order-dir                 | `cifar10-cscores-orig-order.npz (default)`|
|                             | `cifar100-cscores-orig-order.npz`         |
|                             | `cifar10.order.pth cifar100.order.pth ` |
|                             | `cifar100N02.order.pth  cifar100N04.order.pth ` |
|                             | `cifar100N06.order.pth  cifar100N08.order.pth ` |
| --optimizer                 | `sgd(default) nesterov_sgd rmsprop adam` |
| --scheduler                 | `cosine(default) step2 exponential`      |




### Sample codes for Section 2 and Section 3
We provide the sample code for reimplementing the experiments for Figure 2 in Section 2:IMPLICIT CURRICULA 
```
mkdir record
python learnT_order.py --logdir record
```

We provide the sample code for reimplementing the experiments for Figure 3 in Section 3: loss-based c-score. 
```
python loss-based-cscore.py --logdir orders
```
For more information on this code, please see the '.orders/README.md'

### Disclaimer
This is not an official Google product.

