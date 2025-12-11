from NLSD.curve_fit import curve_fit, curve_fit_nele
from MNIST.mnist import mnist_train, mnist_eval, mnist_train_nele, mnist_train_study
from MNIST_ENC.mnist_enc import mnist_enc_train, mnist_enc_eval, mnist_enc_train_nele, mnist_enc_train_study
from NLSD.post_process_nld import process_nld
from MNIST.post_process_tsinm import process_mnist
from MNIST_ENC.post_process_tsinm import process_mnist_enc_study
from MNIST_ENC.post_process_mnist import process_mnist_enc
from MNIST.post_process_mnist_nele import mnist_nele
from MNIST_ENC.post_process_mnist_nele import mnist_enc_nele
from CIFAR10.post_process_cifar10 import process_cifar10
from CIFAR10.post_process_cifar10_nele import process_cifar10_nele
from CIFAR10.cifar import cifar10_train, cifar10_train_nele
import argparse

if __name__ == "__main__":
    print("Begin Analysis")
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default='nlsd')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--reset", type=int, default=0)
    parser.add_argument("--exec", type=int, default=0)
    
    args = parser.parse_args()
    if args.type == 'nlsd':
        if args.mode == 'train':
            curve_fit(args.device, args.reset, args.exec)
        elif args.mode == 'train_nele':
            curve_fit_nele(args.device, args.reset, args.exec)
        elif args.mode == 'train_study':
            curve_fit(args.device, args.reset, 1.0)
        elif args.mode == 'process':
            process_nld(args.reset)
    elif args.type == 'mnist':
        if args.mode == 'train':
            mnist_train(args.device, args.reset, args.exec)
        elif args.mode == 'train_study':
            mnist_train_study(args.device, args.reset, args.exec)
        elif args.mode == 'train_nele':
            mnist_train_nele(args.device, args.reset, args.exec)
        elif args.mode == 'eval':
            mnist_eval(args.device, args.reset)
        elif args.mode == 'process':
            process_mnist()
        elif args.mode == 'process_nele':
            mnist_nele(args.device, args.reset)
    elif args.type == 'mnistenc':
        if args.mode == 'train':
            mnist_enc_train(args.device, args.reset, args.exec)
        elif args.mode == 'train_study':
            mnist_enc_train_study(args.device, args.reset, args.exec)
        elif args.mode == 'train_nele':
            mnist_enc_train_nele(args.device, args.reset, args.exec)
        elif args.mode == 'eval':
            mnist_enc_eval(args.device, args.reset)
        elif args.mode == 'process_study':
            process_mnist_enc_study()
        elif args.mode == 'process':
            process_mnist_enc()
        elif args.mode == 'process_nele':
            mnist_enc_nele(args.device, args.reset)
    elif args.type == 'cifar10':
        if args.mode == 'train':
            cifar10_train(args.device, args.reset, args.exec)
        if args.mode == 'train_nele':
            cifar10_train_nele(args.device, args.reset, args.exec)
        elif args.mode == 'process':
            process_cifar10()
        elif args.mode == 'process_nele':
           process_cifar10_nele()
    print("Analysis completed")