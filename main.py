from NLSD.curve_fit import curve_fit
from MNIST.mnist import mnist_train, mnist_eval
from NLSD.post_process_nld import process_nld
from MNIST.post_process_mnist import process_mnist
from MNIST.post_process_mnist_nele import mnist_nele
from CIFAR10.post_process_cifar10 import process_cifar10
from CIFAR10.post_process_cifar10_nele import process_cifar10_nele
from CIFAR10.cifar import cifar10_train
import argparse

if __name__ == "__main__":
    print("Begin Analysis")
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default='nlsd')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--reset", type=int, default=0)
    
    args = parser.parse_args()
    if args.type == 'nlsd':
        if args.mode == 'train':
            curve_fit(args.device, args.reset)
        elif args.mode == 'process':
            process_nld()
    elif args.type == 'mnist':
        if args.mode == 'train':
            mnist_train(args.device, args.reset)
        elif args.mode == 'eval':
            mnist_eval(args.device, args.reset)
        elif args.mode == 'process':
            process_mnist()
        elif args.mode == 'process_nele':
            mnist_nele()
    elif args.type == 'cifar10':
        if args.mode == 'train':
            cifar10_train(args.device, args.reset)
        elif args.mode == 'process':
            process_cifar10()
        elif args.mode == 'process_nele':
           process_cifar10_nele()
    print("Analysis completed")