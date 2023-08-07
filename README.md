## MetricGAN-OKD: Multi-Metric Optimization of MetricGAN via Online Knowledge Distillation for Speech Enhancement

This is the official PyTorch implementation for the paper "[MetricGAN-OKD](https://proceedings.mlr.press/v202/shin23b.html): Multi-Metric Optimization of MetricGAN via Online Knowledge Distillation for Speech Enhancement" (ICML 2023).

:bell: We are pleased to announce that MetricGAN-OKD was accepted at **ICML23**. :bell:

## Abstract
In speech enhancement, MetricGAN-based approaches reduce the discrepancy between the $L_p$ loss and evaluation metrics by utilizing a non-differentiable evaluation metric as the objective function.
However, optimizing multiple metrics simultaneously remains challenging owing to the problem of confusing gradient directions. In this paper, we propose an effective multi-metric optimization method in MetricGAN via online knowledge distillation---MetricGAN-OKD.
MetricGAN-OKD, which consists of multiple generators and target metrics, related by a one-to-one correspondence, enables generators to learn with respect to a single metric reliably while improving performance with respect to other metrics by mimicking other generators.
Experimental results on speech enhancement and listening enhancement tasks reveal that the proposed method significantly improves performance in terms of multiple metrics compared to existing multi-metric optimization methods.
Further, the good performance of MetricGAN-OKD is explained in terms of network generalizability and correlation between metrics.


## Main results (SE)

| Target Metric       | PESQ | CSIG | CBAK | COVL | Note |
|---------------------|------|------|------|------|------|
| PESQ, CSIG          | 3.24 | 4.23 | 3.07 | 3.73 |------|
| CSIG, PESQ          | 3.19 | 4.26 | 3.12 | 3.72 |------|
| PESQ, CSIG+CBAK+COVL| 3.15 | 4.26 | 3.25 | 3.71 |------|
| PESQ, CSIG+CBAK     | 3.12 | 4.17 | 3.13 | 3.64 |Causal|


## MetricGAN-OKD for Speech Enhancement
You can find the source code on speech ehnancement in the subfolder [SE/](https://github.com/wooseok-shin/MetricGAN-OKD/tree/master/SE).

## MetricGAN-OKD for Listening Enhancement
You can find the source code on speech ehnancement in the subfolder [LE/](https://github.com/wooseok-shin/MetricGAN-OKD/tree/master/LE).



## Todo List
- [ ] Listening Enhancement implementation (training/testing)