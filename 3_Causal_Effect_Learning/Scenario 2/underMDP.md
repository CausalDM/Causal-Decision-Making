# Markov Decision Processes

In this section, we move to the case where the data is no long single stage. In some real cases, researchers may encounter the longitudinal data with long horizons or even infinite horizons. For example, in clinical trial, the doctor will periodically check the health status of patients to provide them with personalized treatment at each time they visit. Under this scenario, the problem becomes: ``What's the causal effect of taking a target treatment at each stage versus the control?". 

Due to the complexity of data structure in multiple horizons, a substantial amount of work was developed based on the Markov assumption between state transitions, which makes the problem easier to deal with. This problem is often known as the causal effect estimation under Markov Decision Processes (MDPs). Specifically, MDP requires two types of conditional independence assumptions to hold: 

**(MA) Markov assumption**:  there exists a Markov transition kernel $\mathcal{P}$ such that  for any $t\ge 0$, $\bar{a}_{t}\in \{0,1\}^{t+1}$ and $\mathcal{S}\subseteq \mathbb{R}^d$, we have 
$\mathbb{P}\{S_{t+1}^*(\bar{a}_{t})\in \mathcal{S}|W_t^*(\bar{a}_t)\}=\mathcal{P}(\mathcal{S};a_t,S_t^*(\bar{a}_{t-1})).$

**(CMIA) Conditional mean independence assumption**: there exists a function $r$ such that  for any $t\ge 0, \bar{a}_{t}\in \{0,1\}^{t+1}$, we have 
$\mathbb{E} \{Y_t^*(\bar{a}_t)|S_t^*(\bar{a}_{t-1}),W_{t-1}^*(\bar{a}_{t-1})\}=r(a_t,S_t^*(\bar{a}_{t-1}))$.




[TO BE ADDED]

First: make the connection with off-policy evaluation in CPL-Paradigm2. Clearly point out the relationship between CEL and CPL.

Then: talk about the methodologies in:

1. Basic setting when all three assumptions `SUTVA`, `No Unmeasured Confounders` and `Positivity` are hold; What kind of work are involved;

2. Some extensions of MDP, such as high order MDP and POMDP; 

3. Violation of the big three assumptions: interference, unmeasured confounders, and the violation of positivity assumptions. Related work will be cited to give a general picture of what have been done in literature.


