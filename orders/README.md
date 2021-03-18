### The folder is used in "main_w_test.py" or "main_wo_test.py". 
You may run in the parent directory to see how this folder works:
```
python main_w_test.py
```
It will download the data "cifar10-cscores-orig-order.npz" or "cifar100-cscores-orig-order.npz" from
https://github.com/pluskid/structural-regularity/tree/master/docs/cscores 


###### Alternatively, you can mannually download the data as follows

```
wget https://github.com/pluskid/structural-regularity/raw/master/docs/cscores/cifar10-cscores-orig-order.npz
wget https://github.com/pluskid/structural-regularity/raw/master/docs/cscores/cifar100-cscores-orig-order.npz
```


###### In our paper, we use the orderings generated from the code "loss-based-cscore.py" in the parent directory:
```
python loss-based-cscore.py --logdir orders
```
This code creates *cifar10.order.pth*, which stores the ordering of each image from small to large loss with a dictionary format:  `{image-index: [loss value, standard deviation of the loss]}`.

For other orders: `cifar100.order.pth  cifar100N02.order.pth  cifar100N04.order.pth cifar100N06.order.pth  cifar100N08.order.pth`, you need to run in parent directory:  

```
python loss-based-cscore.py --logdir orders --dataset cifar100 # cifar100.order.pth
python loss-based-cscore.py --logdir orders --dataset cifar100N --rand_fraction 0.2  # cifar100N02.order.pth
python loss-based-cscore.py --logdir orders --dataset cifar100N --rand_fraction 0.4  # cifar100N04.order.pth
python loss-based-cscore.py --logdir orders --dataset cifar100N --rand_fraction 0.6  # cifar100N06.order.pth
python loss-based-cscore.py --logdir orders --dataset cifar100N --rand_fraction 0.8  # cifar100N08.order.pth
```
We will release `cifar10.order.pth  cifar100.order.pth  cifar100N02.order.pth  cifar100N04.order.pth cifar100N06.order.pth  cifar100N08.order.pth` in the coming future, stay tuned! 



### Disclaimer
This is not an official Google product.