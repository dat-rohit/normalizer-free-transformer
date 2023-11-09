# python train_cifar10.py --net vit_small --n_epochs 200 --size 224 --patch 16
python train_cifar10.py --net vit_ti --n_epochs 200

wait

python train_cifar10.py --net vit_ti --n_epochs 200 --size 224 --patch 16

wait

python train_cifar10.py --net vit_s --n_epochs 200

wait

python train_cifar10.py --net vit_s --n_epochs 200 --size 224 --patch 16