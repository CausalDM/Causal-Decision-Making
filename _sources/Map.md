# Map

In this section, we provide a structured map for the zoo of off-policy evaluation and optimization methods. 
The main purpose is to guide practitioners to find appropriate solutions for their problems, and hence to reduces the gap from academic research to real-world applications. 
We focus on several key decision points, including action, reward, features, policy classes, confounders, side information, etc. 

**RZ**: given our target audience, I suggest to set our taxonomy as application-oriented instead of method-oriented. 
Let's 

1. Collect some papers
2. Classify them based on the taxonomy and select according the quality
3. Draw the map
4. Begin to review these methods and implement them

We aim to include the cited papers in this tutorial and also add the corresponding Python implementations to the accompanying package, in an incremental manner. 

<!--
3.1.1 parametric: Q-learning, etc.
3.1.2 semiparametric: A, single index, etc.
3.1.3 nonparametric: OWL, tree based, etc. (including review of ML)
-->

## Off-Policy Evaluation

## Off-Policy Optimization

## Statistical Inference

💥The following are examples only
## Supported Offline Algorithms
![Offline.png](Offline.png)
| Algorithm | Treatment Type | Outcome Type | Single Stage? | Multiple Stages? | Infinite Horizon? | Evaluation? | C.I.? | Advantages |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [Q-Learning]() | Discrete | Continuous (Mean) |✅|✅| ❗ TODO |✅|❗ TODO| |
| [A-Learning]() | Discrete | Continuous (Mean) |✅|✅|  |✅|❗ TODO| |
| [OWL]() | Discrete | Continuous (Mean) |✅|| ||| |
| [Quatile-OTR](https://doi.org/10.1080/01621459.2017.1330204) | Discrete | Continuous (Quantiles) |✅|  |  || | |


## Supported Online Algorithms
![Online.png](Online.png)
| algorithm | Action Type | Reward Type | Advantages |
|:-|:-:|:-:|:-:|
| [Multi-Armed Bandit]() |⛔| | |
| [Contextual Multi-Armed Bandit]() |⛔| | |