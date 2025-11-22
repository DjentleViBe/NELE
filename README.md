# NELE : **N**on-uniform rational b-spline **E**lementwise **LE**arnable

This repository is an implementation of the paper.

## Pre-requisites
1. Clone the repository:
```
git clone 
```
2. To install requirements:
```
pip install -r requirements.txt
```

# Studies
There are 3 types of analysis that can be performed:
- Non-linear Synthetic Dataset (NLSD)
- MNIST
- CIFAR-10

The details for each simulation can be set inside [config.py](./config.py)
The device is set to ```cpu```, by default. It can be changed by using the argument, ```--device=cuda```.
## NLSD 
To run the training :
```
python main.py --type=nlsd --mode=train
```
The command runs NLSD cases for the following activation functions (AF) : $Tanh$, $ReLU$, $ELU$, $GELU$, $Sigmoid$, $Leaky-ReLU$, $SiLU$, $Softplus$, $LeLU$, $Mish$, $NELE$.

To run the validation :
```
python main.py --type=nlsd --mode=eval
```
To run post-processing :
```
python main.py --type=nlsd --mode=process
```
The follwing plots are generated:
- Training Loss vs epochs
- Curve fitting different AF to data
- Standard deviation and test loss against data

## MNIST
To run the training :
```
python main.py --type=mnist --mode=train
```
To run the validation :
```
python main.py --type=mnist --mode=eval
```
To run post-processing :
```
python main.py --type=mnist --mode=process
```
To run sensitivity analysis :
```
python main.py --type=mnist --mode=process-nele
```
This command can be used for processing different configurations with $NELE$. Plots will contain data for cases defined inside [```AF_plot```](./config.py) array.

## config.py
The ```config.py``` is divided into sections depending on the daata used for the analysis.
### AF
This array holds variables required to indicate AF type to be used for the simulation. A detailed list of supported variables and their AF is given below:
| Variable name | AF | 
|---------------|-----------------|
tanh | Tanh
relu |ReLU
elu|ELU 
gelu |GELU 
sigmoid|Sigmoid
leaky_relu|Leaky_ReLU
silu|SiLU
softplus|Softplus
lelu|LeLU
mish |Mish 
nele|NELE

To facilitate running studies with different configurations, the variables can be appended with additional text followed by ```=``` such as ```nele=0.001```. ```AF_plot``` array also needs to be updated accordingly if the cases need to be post-processed.
A folder with this name is created inside ```RESULTS```, which holds the simulation raw files and ```PICS```, which contains any post-processing files.

