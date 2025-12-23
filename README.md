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

## Settings
[config](./config.py) for different studies can be downloaded from the releases section
## Additional figures
AF approximation plots are provided inside [figures](./figures)

## NLSD 
```NELE``` Parameters used for NLSD curve fitting study. These need to be changed inside [config.py](./config.py) in order to reproduce the results of the paper.
|Parameter|Exp|Hyp|Quad|Sine|Trig|Poly|
|---|---|---|---|---|---|---|
|w0|1.9807|0.5|0.5|1.9807|1.9807|1.9807|
|w1|0.7178|1.0|1.5|0.7178|0.7178|1.5|
|w2|1.0|1.0|1.0|0.1238|1.5|1.0|
|w3|1.5|1.0|1.0|1.5|0.1981|0.1981|
|l|-0.5|-1|-0.5|-1|-1|-1|
|cp0_x|-1.0|-1.0|-1.0|-1.0|-1.0|-1.0|
|cp1|[-0.3, -0.1]| [-0.1, -0.1]|[-0.1, -0.3]|[-0.1, -0.1]|[-0.3, -0.1]|[-0.1, -0.3]|
|Learnable|False|True|False|True|False|True|
|Masking|2|1|2|1|2|1|
|Clamping|True|False|True|False|True|False|

## MNIST

```NELE``` Parameters. This can be changed in [config.py](./config.py)

|Parameter|Value|
|----|-----|
w0 | 1.4205
w1 | 1.004
w2 | 0.3203
w3 | 0.2403
cp0 | [-2.8346, 0.0]
cp1 | [-0.1468, -0.1846]
cp2 | [-0.6443/1.4142, -0.6443/1.4142]
cp3 | [0.0, 0.0]
|Learnable|False|
|Masking|2|
|Clamping|True|
|Learning Rate|0.0014|

## MNIST autoencoder

```NELE``` Parameters. This can be changed in [config.py](./config.py)

|Parameter|Value|
|----|-----|
w0 | 1.8427
w1 | 0.8249
w2 | 1.0931
w3 | 0.3174
cp0 | [-5.4016, 0.0]
cp1 | [-0.1387, -0.1328]
cp2 | [-0.9417/1.4142, -0.9417/1.4142]
cp3 | [0.0, 0.0]
|Learnable|False|
|Masking|2|
|Clamping|True|

## CIFAR10

```NELE``` Parameters. This can be changed in [config.py](./config.py)

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
|Learnable|False|
|Masking|2|
|Clamping|True|

## CIFAR100

```NELE``` Parameters. This can be changed in [config.py](./config.py)

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
|Learnable|False|
|Masking|2|
|Clamping|True|

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
