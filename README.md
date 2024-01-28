# CARS
Code for Experiments done in the paper [Curvature-Aware Derivative-Free Optimization](https://arxiv.org/abs/2109.13391).

This repository includes three folders for three different experiments.

1. CARS_minimal:
  This includes the minimal example for CARS/CARS-NQ in MATLAB

2. CARS_MoreGarbowHillstrom: 
  This experiment solves More-Garbow-Hillstrom problems with CARS and some other algorithms for comparison, and produces a performance profile.
  
3. CARS_MNIST: 
  This is for the black-box adversarial attack on MNIST dataset.

Each folder has its own Readme file, so have a look at it for more details.

## Link to the Refactored Version
There is a refactored version of this repo: [CARS Refactored](https://github.com/bumsu-kim/CARS_Refactored).

In this version, you can simply do, for instance,
```python
from cars.util import setup_default_optimizer

opt = setup_default_optimizer("CARS", f = my_func, x0 = x0)
opt.optimize()
```
or easily fine-tune the parameters for the optimizer using a configuration json file.

Interested readers can read [this](https://github.com/bumsu-kim/CARS_Refactored/blob/master/README.md).