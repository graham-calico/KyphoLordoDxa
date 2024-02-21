# Measurement of Kyphosis and Lordosis from Dexa Scans

![](https://github.com/calico/myproject)

## Overview

This tool applies ML models to the analysis of DEXA images for measuring
curvature at the upper (thoracic) spine (kyphosis), or bottom (lumbar) spine (lordosis).

## [Kyphosis/Lordosis: Methods Description](docs/analysis.md)

## [Developer Documentation](docs/developer.md)

## Installation
The recommended build environment for the code is to have [Anaconda](https://docs.anaconda.com/anaconda/install/) installed and then to create a conda environment for python 3 as shown below:

```
conda create -n kypho python=3.7
```

Once created, activate the environment and install all the needed libraries as follows: 

``` 
conda activate kypho
pip install -r requirements.txt
```

## Usage 
An example for a recommended invokation of the code, to measure kyphosis:

```
python spineCurve.py -i <dir of imgs> -o <out file> --aug_flip --aug_tilt 0.5
```
...or to measure lordosis:
```
python spineCurve.py -i <dir of imgs> -o <out file> --aug_flip --aug_tilt 0.5 --lumbar
```
### [Detailed Usage Instructions](docs/getstarted.md)


## License

See LICENSE
