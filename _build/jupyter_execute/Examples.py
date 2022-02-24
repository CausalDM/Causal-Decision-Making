#!/usr/bin/env python
# coding: utf-8

# ## Motivating Examples
# 
# ### Personalized Incentives
# User growth and engagement are critical in a fast-changing market. Marketing campaigns in internet companies offer quantifiable incentives to encourage users to engage or use new products. The treatment has positive effects on desired business growth, also lead to a surplus in operation cost. The increasing cost impels internet companies to carry out more refined strategies in user acquisition, user retention, and etc. Specifically, the associated costs of massive-scale promotion campaigns must be balanced by incremental business value, with a sustainable return-on-investment (ROI). We are required to predict, for each user, the change in business value caused by different incentive actions, in order to maximize the ROI of market campaigns. This problem is known as uplift modeling, or heterogeneous treatment effect estimation, which has received more and more attention in the causal inference literature.
# 
# ### Ad Targeting & Bidding
# John Wanamaker once phrased 'Half the money I spend on advertising is wasted, the trouble is I don't know which half.'
# It indicates that we are wasting our advertisement on users with high intention to convert naturally. Today's digital technique enables us to estimate the conversion lift of each user via randomized controlled studies. We randomly select users to form two groups, one can be intervened via our ads and the other cannot. Based on the collected data, we can estimate the difference of conversion between ad intervention and no ad intervention. We can then target users with high converion lift, or even increase our bidding price to win the impression for those users.
# 
# 
# ### Multi-touch Attribution
# Users may interact with the same advertisement (possibly with different styles) many times through different channels. 
# Multi-touch attribution, which allows distributing the credit to all related advertisements based on their corresponding contributions, has recently become an important research topic in digital advertising. Rules based on simple intuition have been used in practice for a long time. With the ever enhanced capability to tracking advertisement and users’ interaction with the advertisement, data-driven multi-touch attribution models, which attempt to infer the contribution from user interaction data, become increasingly more popular recently. This problem can be easily formalized to the multi-stage dynamic treatment regime framework. 
# 
