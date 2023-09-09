# Overview
## Introduction
---



## What to expect?
---

```{figure} pics/Workflow_CausalDM.png
---
width: 800px
align: center
---
Workflow of the Causal Decision Making.
```

The Fig 1 depicts the overall structure of this book, which is comprised of three primary components: **Causal Structure Learning**, **Causal Policy Learning**, and **Causal Effect Learning**. Specifically, in the chapter [**Causal Structure Learning (CSL)**](#SL), we present state-of-the-art techniques for learning the skeleton of causal relationships among input variables. When a causal structure is known, the second chapter [**Causal Effect Learning (CEL)**](#ML) introduces approaches for treatment effect identification, estimation and inference. Finally, the [**Causal Policy Learning (CPL)**](#PL) chapter introduces diverse policy learners to learn optimal policies and evaluate various policies of interest.

Following is a brief summary of the contents of each chapter.

## <a name="SL"></a> Causal Structure Learning (CSL)
---
The main goal of causal structure learning is to learn the unknown causal relationships among different variables.

```{image} pics/CSL_aim.png
:alt: Scenario1
:width: 500px
:align: center
```

The classical causal structure learning methods can be categorized into three types.

```{image} pics/CSL_type.png
:alt: Scenario1
:width: 500px
:align: center
```

This chapter discusses three classical techniques for learning causal graphs, each with its own merits and downsides.

| Learners      Type    | Supported Model  | Noise Required for Training |   Complexity     | Scale-Free? | Learners Example |
|-----------------------|------------------|-----------------------------|------------------|-------------|------------------|
|    Testing based      |      Models 1    |          Gaussian           |     $O(p^q)$     |     Yes     |         PC       |
|    Functional based   |   Models 1 & 2   |        non-Gaussian         |     $O(p^3)$     |     Yes     |       LiNGAM     |
|    Score based        |   Models 1 & 3   |    Gaussian/non-Gaussian    |     $O(p^3)$     |     No      |       NOTEARS    |

*$p$ is the number of nodes in $\mathcal{G}$, and $q$ is the max number of nodes adjacent to any nodes in $\mathcal{G}$.*

## <a name="ML"></a> Causal Effect Learning (CEL)
---

Causal effect learning, as we've mentioned at the beginning, aims to infer on the effect of a specific treatment in the context of causal inference. According to the data structure, we mainly divide the problem settings of CEL into three categories: independent states, Markovian state transition, and non-Markovian state transition. 



![Table_of_Fixed_Policy.png](pics/Table_of_Fixed_Policy.png)



1. **Independent states**, or Paradigm 1, denotes the single-stage setup where the full data can be summarized as a number of state-action-reward triplet with size $n$, i.e. $(S_i,A_i,R_i)_{1\leq i\leq n}$. Due to the simplicity of data structure, there are quite a few methods proposed to handle the estimation of both average treatment effect and heterogeneous treatment effect, ranging from basic learners to deep-learning related approaches. 
2. **Markovian state transition**, or Paradigm 2, denotes the case where the data contains $T$ stages, i.e.  $(S_{i,t},A_{i,t},R_{i,t})_{1\leq i\leq n,0\leq t\leq T}$, and the transition of stages follows a Markov decision process. 
3. **Non-Markovian state transition**, or Paradigm 3, denotes other miscellaneous cases where the data have a relatively complex data structure, while the transition of stages doesn't follow the Markov assumption. In this section, we will mainly discuss some representative methods to deal with panel data.

### <a name="Case1"></a> Paradigm 1: I.I.D

In Paradigm 1, we consider the standard case where all observations are i.i.d.. The full data of interest is $(S_i,A_i,R_i)$ where $i\in\{1,\dots,N\}$.



```{image} pics/CEL-IID.png
:alt: Scenario1
:width: 500px
:align: center
```



### <a name="Case1"></a> Paradigm 2: Markovian State Transition

In Paradigm 2, the data we observed can be denoted as $(S_{i,t},A_{i,t},R_{i,t})_{1\leq i\leq n,0\leq t\leq T}$, where $n$ is the number of trajectories, and $T$ is the number of stages. This data structure is widely named as Markov decision processes (MDPs). 

In causal effect learning, we focus on estimating the difference of the effect between a specific target policy (or treatment) and control at all stages. Due to the long-stage or even infinite-horizon structure of the data, most of the existing approaches paid attention to evaluating the expected reward of any given policy (treatment), and then do subtraction to obtain the effect of treatment versus control. 

In observational data analysis, the data we obtained does not come from the target policy (or treatment) we wish to evaluate, resulting in the shift of data distribution. This problem is widely known as offline policy evaluation (OPE) under MDPs. The figure below depicts several groups of methods to address this problem.



```{image} pics/CEL-Markovian.png
:alt: Scenario2
:width: 500px
:align: center
```



Since this problem can be regarded as a special case of causal policy learning, we leave the detailed introduction of this part to Paradigm 2 of chapter 3 (Causal Policy Learning).



### <a name="Case1"></a> Paradigm 3: Panel Data

In Paradigm 3, we consider the panel data where samples are measured over time. This type of data can be found in economics, social sciences, medicine and epidemiology, finance, and the physical sciences. The outcome of interest is defined as $R_{i,t}$, which denotes the reward of observation $i$  at time $t$. 

Consider a (either experimental or observational) study with $N = m + n$ units and $T = T_0 + T_1$ time periods in total. Without loss of generality, we assume that the first m units are treated units and the last n units are control units. Each unit $i$ is associated with a $d$-dimensional time-invariant feature vector $S_i\in \mathbb{R}^d$,  and receives an unit-level outcome $R_{i,t}$ at time $t$. The full data structure is given below:


$$
\left[
\begin{array}{ccc:ccc}
S_{1} & \cdots & S_{m} & S_{m+1} & \cdots & S_{n} \\
\hline
R_{1,1} & \cdots & R_{m,1} & R_{m+1,1} & \cdots & R_{m+n,1} \\
\vdots & & \vdots &\vdots & & \vdots \\
R_{1,T_0} & \cdots & R_{m,T_0} & R_{m+1,T_0} & \cdots & R_{m+n,T_0} \\
\hdashline
R_{1,T_0+1} & \cdots & R_{m,T_0+1} & R_{m+1,T_0+1} & \cdots & R_{m+n,T_0+1} \\
\vdots & & \vdots &\vdots & & \vdots \\
R_{1,T} & \cdots & R_{m,T} & R_{m+1,T} & \cdots & R_{m+n,T} \\
\end{array}
\right]
$$

The current literature in dealing with panel data can be roughly divided into two categories: Difference-in-difference and synthetic control. 



```{image} pics/CEL-PanelData.png
:alt: Scenario3
:width: 500px
:align: center
```


In general, DiD methods are often applied in cases where the number of treated units and control units are comparable. The entire methodology is based on a key assumption, which is well known as "common trend" (CT) or "bias stability" (BS) assumption. This assumption guarantees that the expected changes in the potential outcome over time are unrelated to belonging to treatment or control group. In contrast, Synthetic Control (SC) methods require fewer assumptions, believing that a weighted average of control units can
provide a good approximation for the counterfactual outcome of the treated unit as if it has been under control.





## <a name="PL"></a> Causal Policy Learning (CPL)

---
This chapter focuses on six common data dependence structures in decision making, including [**Fixed Policy with Independent States**](#Case1), [**Fixed Policy with Markovian State Transition**](#Case2), [**Fixed Policy with Non-Markovian State Transition**](#Case3), [**Adaptive Policy with Independent States**](#Case4), [**Adaptive Policy with Markovian State Transition**](#Case5), and [**Adaptive Policy with Non-Markovian State Transition**](#Case6). The similarities and differences between six paradigms are summarized as follows.

![Table_of_Six_Scenarios.png](pics/Table_of_Six_Scenarios.png)

### <a name="Case1"></a> Paradigm 1: Fixed Policy with Independent States
As the figure illustrated, observations in Paradigm 1 are i.i.d. sampled. For each observation, there are three components, $S_i$ is the context information if there is any, $A_i$ is the action taken, and $R_i$ is the reward received. When there is contextual information, the action would be affected by the contextual information, while the final reward would be affected by both the contextual information and the action. A classical class of problems that are widely studied in this context is the **Single-Stage Dynamic Treatment Regime (DTR)**[1]. In this book, we mainly focus on methods for policy evaluation and policy optimization for Single-Stage DTR, with a detailed map in [Appendix A](#SingleDTR)

### <a name="Case2"></a> Paradigm 2: Fixed Policy with Markovian State Transition
The Paradigm 2 is well-known as Markov Decision Process (MDP), whose main characteristic is the Markovian state transition. In particular, while $A_t$ is only affected by $S_t$, both $R_t$ and $S_{t+1}$ would be affected by $(S_t,A_t)$. Given $S_{t}, A_t$, a standard assumption of MDP problems is that $R_t$ and $S_{t+1}$ are independent of previous observations. A list of related learning methods will be introduced, with a map in [Appendix B](#MDP).

### <a name="Case3"></a> Paradigm 3: Fixed Policy with Non-Markovian State Transition
When a history-independent policy is applied, the Paradigm 3 takes all the possible causal relationships into account and is well-known as the multiple-stage DTR problem [1]. 
In this book, we introduce two classical learning methods, including Q-learning and A-learning (See a map in [Appendix C](#MultiDTR)). 


### <a name="Case4"></a> Paradigm 4: Adaptive Policy with Independent States
The Paradigm 4 setting is widely examined in the online decision making literature, especially the bandits, where the treatment policy is time-adaptive. Specifically, $H_{t-1}$ includes all the previous observations up to time $t-1$ (include observations at time $t-1$) and is used to update the action policy at time $t$, and therefore affect the action $A_t$. While $S_t$ is i.i.d sampled from the correponding distribution, $R_t$ is influenced by both $A_t$ and $S_t$. Finally, the new observation $(S_t,A_t,R_t)$, in conjunction with all previous observations, would then be formulated as $H_{t+1}$ and affect $A_{t+1}$ only. A structure that lacks contextual information $S_t$ is also very common. In this book, a list of bandits algorithms would be introduced, with a detailed map in [Appendix D](#Bandits).


### <a name="Case5"></a> Paradigm 5: Adaptive Policy with Markovian State Transition
Building upon the MDP structure, when an adaptive policy is applied, the Scenario 5 clearly depicts the data-generating process, in which  $S_t$ follows the Markovian state transition and $A_t$ would be affected by all previous observations $H_{t-1}$. 
This corresponds to the typical online RL setup. 


### <a name="Case6"></a> Paradigm 6: Adaptive Policy with Non-Markovian State Transition
We can further relax the Markovian assumption required in Paradigm 5 to allow Non-Markovian State Transition, which includes the DTR bandits and Partially Observable Markov Decision Processes (POMDP) problems. 


**Extensions.** Along the y-axis, we can further consider the case where the data collection policy depends on some unobservable variables, which correspond to the *confounded* problems. 


## Appendix
---
### <a name="SingelDTR"></a> A. Paradigm 1
```{image} pics/CPL_Paradigm1.png
:alt: Scenario1
:width: 500px
:align: center
```

| Algorithm | Treatment Type | Outcome Type | Single Stage? | Multiple Stages? | Infinite Horizon? | Evaluation? | Optimization? | C.I.? | Advantages |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [Q-Learning](https://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf) | Discrete | Continuous (Mean) |✅|✅| |✅|✅| ||
| [A-Learning](https://www.researchgate.net/profile/Eric-Laber/publication/221665211_Q-_and_A-Learning_Methods_for_Estimating_Optimal_Dynamic_Treatment_Regimes/links/58825d074585150dde402268/Q-and-A-Learning-Methods-for-Estimating-Optimal-Dynamic-Treatment-Regimes.pdf) | Discrete | Continuous (Mean) |✅|✅|  |✅|✅| ||
| [OWL](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2012.695674?casa_token=bwkVvffpyFcAAAAA:hlN58Fbz59blLj5npZFjEQD-HkPeMevEN5pWWLu_vuIVxPWl5aYShgCVHUVeODAfj6Pr8DpzGFlPZ1E) | Discrete | Continuous (Mean) |✅|❗BOWL| |✅|❗TODO| ||
| [Quatile-OTR](https://doi.org/10.1080/01621459.2017.1330204) | Discrete | Continuous (Quantiles) |✅✏️|  |  |✅✏️|✅✏️| ||
| [Deep Jump Learner](https://proceedings.neurips.cc/paper/2021/file/816b112c6105b3ebd537828a39af4818-Paper.pdf) | Continuous | Continuous/Discrete |✅|  |  |✅| ✅| | Flexible to implement & Fast to Converge|
| Kernel-Based Learner | | ||  |  ||| ||
| Outcome Learning | | ||  |  ||| ||


### <a name="MDP"></a> B. Paradigm 2
```{image} pics/CPL_Paradigm2.png
:alt: Scenario2
:width: 400px
:align: center
```

### <a name="MultiDTR"></a> C. Paradigm 3
```{image} pics/CPL_Paradigm3.png
:alt: Scenario3
:width: 300px
:align: center
```

| Algorithm | Treatment Type | Outcome Type | Evaluation? | Optimization? | C.I.? | Advantages |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| [Q-Learning](https://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf) | Discrete | Continuous (Mean) |✅|✅| ||
| [A-Learning](https://www.researchgate.net/profile/Eric-Laber/publication/221665211_Q-_and_A-Learning_Methods_for_Estimating_Optimal_Dynamic_Treatment_Regimes/links/58825d074585150dde402268/Q-and-A-Learning-Methods-for-Estimating-Optimal-Dynamic-Treatment-Regimes.pdf) | Discrete | Continuous (Mean)  |✅|✅| ||


### <a name="Bandits"></a> D. Paradigm 4
```{image} pics/CPL_Paradigm4.png
:alt: Scenario4
:width: 900px
:align: center
```

| algorithm | Reward | with features? | Advantage |
|:-|:-:|:-:|:-:|
| **Multi-Armed Bandits** || | |
| [$\epsilon$-greedy]() | Binary/Gaussian | |Simple|
| [TS](https://www.ccs.neu.edu/home/vip/teach/DMcourse/5_topicmodel_summ/notes_slides/sampling/TS_Tutorial.pdf) | Binary/Guaasian | | |
| [UCB1](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf) | Binary/Gaussian | | |
| **Contextual Bandits** || | |
| LinTS | [GLM](http://proceedings.mlr.press/v108/kveton20a/kveton20a.pdf)/[Gaussian](http://proceedings.mlr.press/v28/agrawal13.pdf) | ✅ | |
| LinUCB | [GLM](http://proceedings.mlr.press/v70/li17c/li17c.pdf)/[Guassian](https://dl.acm.org/doi/pdf/10.1145/1772690.1772758?casa_token=CJjeIziLmjEAAAAA:CkRvgHQNqpy10rzcUP5kx31NWJmgSldd6zx8wZxskZYCoCc8v7EDIw3t3Gk1_6mfurqQTqRZ7fVA) | ✅ | |
| **Meta Bandits** || | |
|Meta-TS|| | |
|MTSS|| | |
| **Structured Bandits** || | |
| *Learning to Rank* || | |
| [TS-Cascade](http://proceedings.mlr.press/v89/cheung19a/cheung19a.pdf) | Binary | | |
| [CascadeLinTS](https://arxiv.org/pdf/1603.05359.pdf) | Binary | ✅ | |
| [MTSS-Cascade](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |
| *Combinatorial Optimization* || | |
| [CombTS](http://proceedings.mlr.press/v80/wang18a/wang18a.pdf) | | | |
| [CombLinTS](http://proceedings.mlr.press/v37/wen15.pdf) | | ✅ | |
| [MTSS-Comb](https://arxiv.org/pdf/2202.13227.pdf) | Continuous | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |
| *Assortment Optimization* || | |
| [MNL-Thompson-Beta](http://proceedings.mlr.press/v65/agrawal17a/agrawal17a.pdf) | Binary | | |
| [TS-Contextual-MNL](https://proceedings.neurips.cc/paper/2019/file/36d7534290610d9b7e9abed244dd2f28-Paper.pdf) | Binary | ✅ | |
| [MTSS-MNL](https://arxiv.org/pdf/2202.13227.pdf) | Binary | ✅ | Scalable, Robust, accounts for inter-item heterogeneity |



## Reference
[1] Tsiatis, A. A., Davidian, M., Holloway, S. T., & Laber, E. B. (2019). Dynamic treatment regimes: Statistical methods for precision medicine. Chapman and Hall/CRC.
