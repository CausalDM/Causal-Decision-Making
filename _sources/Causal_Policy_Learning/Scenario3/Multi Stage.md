# Multiple Stages (DTR)

## Problem Setting
Suppose we have a dataset containning observations from $N$ individuals, and each individual has $T$ observations. For each individual $i$, the observed data is structured as follows

    \begin{align}
    (S_{1i},A_{1i},\cdots,S_{Ti},A_{Ti},R_i), i=1,\cdots, N.
    \end{align} 
    Let $h_{ti}=\{S_{1i},A_{1i},\cdots,S_{ti}\})$ includes all the information observed till step t, $\bar{W}_t=(W_1, W_2, \cdots, W_t)$ for any variable $W$, and $R(\bar{a}_T)$ be the potential reward that would be observed if the action sequence $\bar{a}_T$ is taken.


## Real Data
datamdp from the website of DTR