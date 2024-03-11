| Algorithm | Treatment Type | Outcome Type | Single Stage? | Multiple Stages? | Infinite Horizon? | Evaluation? | Optimization? | C.I.? | Advantages |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [Q-Learning](https://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf) | Discrete | Continuous (Mean) |✅|✅| |✅|✅| ||
| [A-Learning](https://www.researchgate.net/profile/Eric-Laber/publication/221665211_Q-_and_A-Learning_Methods_for_Estimating_Optimal_Dynamic_Treatment_Regimes/links/58825d074585150dde402268/Q-and-A-Learning-Methods-for-Estimating-Optimal-Dynamic-Treatment-Regimes.pdf) | Discrete | Continuous (Mean) |✅|✅|  |✅|✅| ||
| [OWL](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2012.695674?casa_token=bwkVvffpyFcAAAAA:hlN58Fbz59blLj5npZFjEQD-HkPeMevEN5pWWLu_vuIVxPWl5aYShgCVHUVeODAfj6Pr8DpzGFlPZ1E) | Discrete | Continuous (Mean) |✅|❗BOWL| |✅|❗TODO| ||
| [Quatile-OTR](https://doi.org/10.1080/01621459.2017.1330204) | Discrete | Continuous (Quantiles) |✅✏️|  |  |✅✏️|✅✏️| ||
| [Deep Jump Learner](https://proceedings.neurips.cc/paper/2021/file/816b112c6105b3ebd537828a39af4818-Paper.pdf) | Continuous | Continuous/Discrete |✅|  |  |✅| ✅| | Flexible to implement & Fast to Converge|
| Kernel-Based Learner | | ||  |  ||| ||
| Outcome Learning | | ||  |  ||| ||


| Algorithm | Treatment Type | Outcome Type | Evaluation? | Optimization? | C.I.? | Advantages |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| [Q-Learning](https://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf) | Discrete | Continuous (Mean) |✅|✅| ||
| [A-Learning](https://www.researchgate.net/profile/Eric-Laber/publication/221665211_Q-_and_A-Learning_Methods_for_Estimating_Optimal_Dynamic_Treatment_Regimes/links/58825d074585150dde402268/Q-and-A-Learning-Methods-for-Estimating-Optimal-Dynamic-Treatment-Regimes.pdf) | Discrete | Continuous (Mean)  |✅|✅| ||


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
