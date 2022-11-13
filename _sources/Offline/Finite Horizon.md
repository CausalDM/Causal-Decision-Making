# Finite Horizon (DTR)

## Problem Setting
Suppose we have a dataset containing observations from $N$ individuals. For each individual $i$, we have $\{\mathbf{X}_{i},A_{i},Y_{i}\}$, $i=1,\cdots,N$. $\mathbf{X}_{i}$ includes the feature information, $A_{i}$ is the action taken, and $Y_{i}$ is the observed reward received.

## Real Data
**1. Fetch_hillstrom**

Fetch_hillstrom is a dataset from the scikit-uplift package with a single decision point. This study aims to assess the effectiveness of an e-mail campaign ($A$). The primary outcome of the interest is the money spent by customers during the two weeks following the e-mail campaign. By selecting only the individuals who purchased merchandise (spend>0), the dataset we used in this chapter comprises 578 customers ($N=578$). For each customer, there are nine features available: 
- **recency**: # of months since the last purchase;
- **history**: total dollars spent in the past year; 
- **mens**: binary, =1 if purchased Men's merchandise in the past year;
- **womens**: binary, =1 if purchased Women's merchandise in the past year;
- **newbie**: binary, =1 if the customer is a new customer in the past year; 
- **zip_code_Surburban**: binary, =1 if zip_code is classified as Suburban; 
- **zip_code_Urban**: binary, =1 if zip_code is classified as Urban;
- **channel_Phone**: binary, =1 if the customer purchased from Phone in the past year;
- **channel_Web**: binary, =1 if the customer purchased from Web in the past year.

Note, if **zip_code_Surburban** =0 and **zip_code_Urban**=0, then the zip_code is classified as Rural; if **channel_Phone**=0 and **channel_Web**=0, then the customer purchased from multichannel in the past year.

There are two different types of action space that are available for us to specify:
- **Binary Treatment**:
Considering a binary action space, each customer would either receive an e-mail campaign ($A=1$) or receive no e-mail ($A=0$).

- **Multi Treatments**:
Considering a multinomial action space, each customer would either receive no e-mail ($A=0$) or receive an e-mail campaign featuring Women's merchandise ($A=1$) or receive an e-mail campaign featuring Men's merchandise ($A=2$)

After two weeks following the e-mail campaign, each customer's total dollar spent ($Y$) is recorded.

The observed data are independent and identically distributed
$\{\text{recency}_i, \text{history}_i, \text{mens}_i, \text{womens}_i, \text{newbie}_i, \text{zip_code_Surburban}_i, \text{zip_code_Urban}_i, \text{channel_Phone}_i,\text{channel_Web}_i ,A_i, Y_i\}$, for $i=1,…,N$,

where larger values of $Y$ are considered better.

- **How to get the data?**
    1. from causaldm._util_causaldm import *
    2. if binary treatment, X,A,Y = get_data(target_col = 'spend', binary_trt = True); 
        otherwise, X,A,Y = get_data(target_col = 'spend', binary_trt = Flase)

More details about the original dataset can be found in [https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).


## Multiple Stages

## Problem Setting
Suppose we have a dataset containning observations from $N$ individuals. For each individual $i$, the observed data is structured as follows
    \begin{align}
    (X_{1i},A_{1i},\cdots,X_{Ti},A_{Ti},Y), i=1,\cdots, N.
    \end{align} 
    Let $h_{ti}=\{X_{1i},A_{1i},\cdots,X_{ti}\})$ includes all the information observed till step t. 

## Real Data