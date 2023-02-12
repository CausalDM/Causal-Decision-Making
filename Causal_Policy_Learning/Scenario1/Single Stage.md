# Single Stage (DTR)

## Problem Setting
Suppose we have a dataset containing observations from $N$ individuals. For each individual $i$, we have $\{\mathbf{S}_{i},A_{i},R_{i}\}$, $i=1,\cdots,N$. $\mathbf{S}_{i}$ includes the feature information, $A_{i}$ is the action taken, and $R_{i}$ is the observed reward received. Further, let $R_i(a)$ denote the potential reward that would be observed if individual $i$ was treated with action $a$.

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

After two weeks following the e-mail campaign, each customer's total dollar spent ($R$) is recorded.

The observed data are independent and identically distributed
$\{\text{recency}_i, \text{history}_i, \text{mens}_i, \text{womens}_i, \text{newbie}_i, \text{zip_code_Surburban}_i, \text{zip_code_Urban}_i, \text{channel_Phone}_i,\text{channel_Web}_i ,A_i, R_i\}$, for $i=1,…,N$,

where larger values of $R$ are considered better.

- **How to get the data?**
    1. from causaldm._util_causaldm import *
    2. if binary treatment, S,A,R = get_data(target_col = 'spend', binary_trt = True); 
        otherwise, S,A,R = get_data(target_col = 'spend', binary_trt = Flase)

More details about the original dataset can be found in [https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).