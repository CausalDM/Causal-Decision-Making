# Markov Decision Processes -- Paradigm 2

In this section, we move to the case where the data is no long single stage. In some real cases, researchers may encounter the longitudinal data with long horizons or even infinite horizons. For example, in clinical trial, the doctor will periodically check the health status of patients to provide them with personalized treatment at each time they visit. Under this scenario, the problem becomes: ``What's the causal effect of taking a target treatment at each stage versus the control?". 

Due to the complexity of data structure in multiple horizons, a substantial amount of work was developed based on the Markov assumption between state transitions, which makes the entire structure easier to deal with. This problem is often known as the causal effect estimation under Markov Decision Processes (MDPs). Specifically, MDP requires two types of conditional independence assumptions to hold: 

**(MA) Markov assumption**:  there exists a Markov transition kernel $\mathcal{P}$ such that  for any $t\ge 0$, $\bar{a}_{t}\in \{0,1\}^{t+1}$ and $\mathcal{S}\subseteq \mathbb{R}^d$, we have 
$\mathbb{P}\{S_{t+1}^*(\bar{a}_{t})\in \mathcal{S}|W_t^*(\bar{a}_t)\}=\mathcal{P}(\mathcal{S};a_t,S_t^*(\bar{a}_{t-1})).$

**(CMIA) Conditional mean independence assumption**: there exists a function $r$ such that  for any $t\ge 0, \bar{a}_{t}\in \{0,1\}^{t+1}$, we have 
$\mathbb{E} \{Y_t^*(\bar{a}_t)|S_t^*(\bar{a}_{t-1}),W_{t-1}^*(\bar{a}_{t-1})\}=r(a_t,S_t^*(\bar{a}_{t-1}))$.


Under Markov decision processes, the data can be written as a state-action-reward tuple where $D=\{S_{i,t},A_{i,t},R_{i,t}\}_{0\leq t\leq T,1\leq i\leq N}$. As an extension of single stage settings, causal effect learning under MDPs aims to estimate the reward difference between treatment $\{A_t\}_{0\leq t\leq T}\equiv 1$ and the control $\{A_t\}_{0\leq t\leq T}\equiv 0$, which are executed at all stages. 

However, setting the action to be all $1$ or all $0$ at all stages is just a special case in sequential decision making, which is limited in real applications. For example, the doctor may assign one kind of treatment to patients at the beginning, while changing to a new treatment according to the health status of patients at later decision points. This yields a more general topic in causal inference: causal policy learning. In causal policy learning, we do not pay specific attention to the reward difference between treatment and control. Instead, we define the value function $V^{\pi}(s)$ as the expected reward one would receive by executing policy (or treatment) $\pi:\mathcal{S}\rightarrow \mathcal{A}$, where
\begin{equation}
\begin{aligned}
V^{\pi}(s) = \sum_{0\leq t\leq T} \gamma^t\mathbb{E}[R_t^*(\pi)|S_0=s],
\end{aligned}
\end{equation}
and $R_t^*(\pi)$ denotes the potential outcome one would receive by executing policy $\pi$.

Therefore, the HTE under MDPs can be derived as $\text{HTE} = V^1(s)-V^0(s)$, and the corresponding ATE is given by $\text{ATE}= \mathbb{E}\{V^1(s)-V^0(s)\}$.

*  **Causal Effect Learning (CEL) under MDPs**: a multiple-stage treatment effect estimation problem, where we are focusing on the reward difference between treatment and control;

*  **Causal Policy Learning (CPL) under MDPs**: not restricted to estimation and offline evaluation; but when talking about offline policy evaluation under MDPs, CEL is a special case of CEL. Most of the existing literature interested in the causal effect in sequential decision making problem will start from the policy evaluation first, and then take the difference to obtain the corresponding treatment effect.

Since CEL under MDPs is a special case of CPL, we will leave the detailed introduction to [CPL-Paradigm 2](https://causaldm.github.io/Causal-Decision-Making/4_Causal_Policy_Learning/Scenario2/preliminary_MDP-potential-outcome.html) for the consistency of content. Specifically, several classical approaches in offline policy evaluation will be elaborated in that section, including fitted Q evaluation, importance sampling estimator under MDPs, double robust estimator under MDPs, etc.  


It is worth noting that all of the above approaches are based on the "big three assumptions" in causal inference (namely, `SUTVA`, `NUC`, `Positivity`), as well as the conditional independence assumptions listed above (i.e. `MA` and `CIMA`). However, these assumptions are often violated especially in observational studies. Recently, significant efforts have been made to relax these conditions. For instance, the interference problem was widely considered in literature when `SUTVA` is violated. Instrumental variables, mediators and proxy variables can be employed to correct bias caused by the unmeasured confounders when `NUC` assumption doesn't hold. Trimming methods  have been proposed to address cases where strict `Positivity` assumption was violated. Moreover, researchers have been exploring high-order MDPs and partially observable MDPs (POMDPs) to relax the strong conditional independence assumptions in MDPs. Although it would be ideal to cover these topics in greater depth, they slightly fall outside the scope of our discussion, and we leave them as potential areas for future work.


