# From Optimization Dynamics to Generalization Bounds via Łojasiewicz Gradient Inequality 


This repository contains the official codes that used for generating numerical results in the following paper:  

Fusheng Liu, Haizhao Yang, Soufiane Hayou, Qianxiao Li, [From Optimization Dynamics to Generalization Bounds via Łojasiewicz Gradient Inequality](https://openreview.net/forum?id=mW6nD3567x), *Transactions on Machine Learning Research*, 2022.



## Generating the numerical results for Figure 1 \& Figure 2

### Saving results


For non-analytic and non-lgi models, modify the function in `get_lgi_counter.py`. Then run the following commands to save results for these two models and linear regression models respectively.
```
python get_lgi_counter.py
python get_lgi_synthetic.py
```

For neural network models, change different models in `get_lgi.py`. Then run the following commands to save results with respect to different models
```
python get_lgi.py
```

### Plotting Figure 1 \& 2

Modifying 'name, K0, s' in `plot_lgi.py`, then
```
python plot_lgi.py
```




## Generating the numerical results for Figure 3

Run the following commands to save results

```
python get_lgi_corruption.py
python get_vary_size.py
```

Then
```
python plot_corruption.py
python plot_vary_size.py
```
