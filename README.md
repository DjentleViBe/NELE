# NELE : **N**on-Uniform Rational B-Spline **E**lementwise **LE**arnable

This repository is an implementation of the paper.

## Pre-requisites
1. Clone the repository:
```
git clone https://github.com/DjentleViBe/NELE
```
2. To install requirements:
```
pip install -e .
```
3. Import the package:
```
from Nele import Nele
```

# Usage

Default parameters : 
```NELE``` parameters provided are as follows:

|Parameter|Value|
|----|-----|
w0 | 1.273
w1 | 0.7607
w2 | 1.4628
w3 | 0.6563
cp0 | [-4.9992, 0.0]
cp1 | [-0.2194, -0.248]
cp2 | [-0.8054/1.4142, -0.8054/1.4142]
cp3 | [0.0, 0.0]
|Learnable|False|
|Masking|2|
|Clamping|True|
