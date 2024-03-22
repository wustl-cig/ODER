# [Online Deep Equilibrium Learning for Regularization by Denoising]

Plug-and-Play Priors (PnP) and Regularization by Denoising (RED) are widely-used frameworks for solving imaging inverse problems by computing fixed-points of operators combining physical measurement models and learned image priors. While traditional PnP/RED formulations have focused on priors specified using image denoisers, there is a growing interest in learning PnP/RED priors that are end-to-end optimal. The recent Deep Equilibrium Models (DEQ) framework has enabled memory-efficient end-to-end learning of PnP/RED priors by implicitly differentiating through the fixed-point equations without storing intermediate activation values.  However, the dependence of the computational/memory complexity of the measurement models in PnP/RED on the total number of measurements leaves DEQ impractical for many imaging applications. We propose ODER as a new strategy for improving the efficiency of DEQ through stochastic approximations of the measurement models. We theoretically analyze ODER giving insights into its convergence and ability to approximate the traditional DEQ approach. Our numerical results suggest the potential improvements in training/testing complexity due to ODER on three distinct imaging applications.

## How to run the code

### Prerequisites for ODER/RED (DEQ)
It is better to first use Conda to set up a new environment by running conda create -n ODER python=3.8.
Then install the python dependencies by running pip install -r requirements.txt.
### Run the Demo

to demonstrate the performance of ODER / RED(DEQ) for Sparse-view CT, you can run the demo code by typing

```
$ python sparseCT.py
```
The per iteration results will be stored in the ./results folder. 
