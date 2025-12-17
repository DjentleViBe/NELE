# NELE : **N**on-Uniform Rational B-Spline **E**lementwise **LE**arnable

This repository is an implementation of the paper.

## Pre-requisites
1. Clone the repository:
```
git clone https://github.com/DjentleViBe/NELE
```
2. To install requirements:
```
pip install -r requirements.txt
```

# Studies
There are 5 types of analysis that can be performed:
- **Non-linear Synthetic Dataset (NLSD)**
- **MNIST**
- **MNIST autoencoder**
- **CIFAR-10**
- **CIFAR-100**

The following activation functions (AF) are considered for comparison : $Tanh$, $ReLU$, $ELU$, $GELU$, $Sigmoid$, $Leaky-ReLU$, $SiLU$, $Softplus$, $LeLU$, $Mish$, $NELE$.
The details for each simulation can be set inside [config.py](./config.py).
The analysis can be launched by running:
```
python main.py --type=<type_id> --mode=<mode_name> --device=<device_name>
```
```<device_name>``` can be changed to ```cpu```, ```cuda``` or ```mps``` depending on hardware.
|<type_id>|Analysis|
|---|---|
| ```nlsd```|NLSD
| ```mnist```|MNIST
| ```mnistenc```|MNIST autoencoder
| ```cifar10```|CIFAR-10
| ```cifar100```|CIFAR-100

|<model_name>|Details|
|---|---|
|```train```| Runs training study across AF for ```epochs``` |
|```train_nele```| Runs training study for ```NELE``` for ```epochs``` |
|```eval```| Runs noise evaluation across AF |
|```process```| Produces Training loss vs epochs, test loss across```epochs``` for different AF |
|```process_study```|Produces median Training loss vs epochs, test loss across```epochs``` for different AF|

```TYPE``` variable inside [config.py](./config.py) can be set ```0```: NELE, ```1```: NELE_PARAM (fixed parameters), ```2```: NELE_LEARN (learnable type)
```3```: NELE_PARAM_DIR (fixed parameters without x minimum rescaling)

## NLSD 
```NELE``` Parameters used for NLSD curve fitting study. These need to be changed inside [config.py](./config.py) in order to reproduce the results of the paper.

|Study|l|w1|w2|y1|x1|y0|```TYPE```|w|cp0|cp1|cp2|cp3|
|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
Exponential|-|-|-|-|-|-|1|[1,1,1,1]|[-1.0, 0.0]|[-0.1, -0.1]| [-1.0/1.4142, -1.0/1.4142]|[0.0, 0.0]
Hyperbola|-1.0|1.0|1.0|-0.1|-0.1|0.0|2|[1,1,1,1]|[-1.0, 0.0]|[-0.1, -0.1]| [-1.0/1.4142, -1.0/1.4142]|[0.0, 0.0]
Quadratic|-1.0|1.0|1.0|-0.1|-0.1|0.0|2|[1,1,1,1]|[-1.0, 0.0]|[-0.1, -0.1]| [-1.0/1.4142, -1.0/1.4142]|[0.0, 0.0]
Sinusoidal|-|-|-|-|-|-|1|[1,1,1,1]|[-4.0, 0.0]|[-0.1, -0.1]| [-1.0/1.4142, -1.0/1.4142]|[0.0, 0.0]
Trigonometric|-|-|-|-|-|-|1|[1,1,1,1]|[-1.0, 0.0]|[-0.1, -0.1]| [-1.0/1.4142, -1.0/1.4142]|[0.0, 0.0]
Polynomial|-1.0|1.0|1.0|-0.1|-0.1|0.0|2|[1,1,1,1]|[-1.0, 0.0]|[-0.1, -0.1]| [-1.0/1.4142, -1.0/1.4142]|[0.0, 0.0]

## MNIST


```NELE``` Parameters. This can be changed in [config.py](./config.py)

```TYPE``` = 1

|Parameter|Value|
|----|-----|
w0 | 1.0
w1 | 1.0
w2 | 1.0
w3 | 1.0
cp0 | [-4.0, 0.0]
cp1 | [-0.1, -0.1]
cp2 | [-1.0/1.4142, -1.0/1.4142]
cp3 | [0.0, 0.0]

## MNIST autoencoder

```NELE``` Parameters. This can be changed in [config.py](./config.py)

```TYPE``` = 1
|Parameter|Value|
|----|-----|
w0 | 1.0
w1 | 1.0
w2 | 1.0
w3 | 1.0
cp0 | [-4.0, 0.0]
cp1 | [-0.1, -0.1]
cp2 | [-1.0/1.4142, -1.0/1.4142]
cp3 | [0.0, 0.0]

## CIFAR10

```NELE``` Parameters. This can be changed in [config.py](./config.py)

```TYPE``` = 4
|Parameter|Value|
|----|-----|
w0 | 1.0
w1 | 1.0
w2 | 1.0
w3 | 1.0
cp0 | [-4.0, 0.0]
cp1 | [-0.1, -0.1]
cp2 | [-1.0/1.4142, -1.0/1.4142]
cp3 | [0.0, 0.0]

## CIFAR100

```NELE``` Parameters. This can be changed in [config.py](./config.py)

```TYPE``` = 4
|Parameter|Value|
|----|-----|
w0 | 1.9807
w1 | 0.7178
w2 | 0.1238
w3 | 0.1981
cp0 | [-4.0, 0.0]
cp1 | [-0.1, -0.1]
cp2 | [-1.0/1.4142, -1.0/1.4142]
cp3 | [0.0, 0.0]

## config.py
```config.py``` is divided into sections depending on the data used for the analysis.
### AF
This array holds variables required to indicate AF type to be used for the simulation. A detailed list of supported variables and their AF is given below:
| Variable name | AF | 
|---------------|-----------------|
tanh | Tanh
relu |ReLU
elu|ELU 
gelu |GELU 
sigmoid|Sigmoid
leaky_relu|Leaky ReLU
silu|SiLU
softplus|Softplus
lelu|LeLU
mish |Mish 
nele|NELE

To facilitate running studies with custom configurations, the variables can be appended with additional text followed by ```=``` such as ```nele=0.001```. ```AF_plot``` array also needs to be updated accordingly if the cases need to be post-processed.
A folder with this name is created inside ```RESULTS```, which holds the simulation raw files and inside ```PICS```, which contains any post-processing files.

