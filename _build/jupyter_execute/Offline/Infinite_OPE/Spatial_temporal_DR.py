#!/usr/bin/env python
# coding: utf-8

# # Infinite Horizon Importance Sampling for Policy Evaluation

# In[1]:


# After we publish the pack age, we can directly import it
# TODO: explore more efficient way
# we can hide this cell later
import os
os.getcwd()
os.chdir('..')
os.chdir('../CausalDM')



# Another important approach is \textit{importance sampling} (IS), also known as inverse propensity score or inverse propensity weighting methods. 
# IS has been widely used in statistics, and the idea can be extended to OPE after appropriately handling the temporal dependency. 
# 
# ## Main Idea
# \subsection{Doubly robust methods}\label{sec:DR}
# The third category, the \textit{doubly robust} (DR) approach, combines DM and IS to achieve low variance and bias \citep{jiang2016doubly, farajtabar2018more, uehara2019minimax}. 
# The DR technique has also been widely studied in statistics \citep{rotnitzky1995semiparametric}. 
# It typically requires both a direct method-type estimator and an IS-type estimator, and then integrates them in a carefully designed form. 
# 
# In OPE, a DR estimator first requires a Q-function estimator, denoted as $\widehat{Q}$, which can be learned by various methods in the literature, such as FQE introduced in Section \ref{sec:DM}. 
# Denote the corresponding plug-in V-function estimator as $\widehat{V}$. 
# These estimators will then be integrated with importance ratios in a form typically motivated by the Bellman equation \eqref{eqn:bellman_Q}. 
# For example, based on the step-wise IS \eqref{eqn:stepIS}, \citet{thomas2016data} proposed to construct the estimator as 
# % \begin{align}\label{eqn:IS}
# %     \hat{\eta}^{\pi}_{IS} = \frac{1}{n} \sum_{i=1}^n \rho^i_T (\sum_{t=0}^{T-1} \gamma^t R_{i,t})
# % \end{align}
# \begin{align*}%\label{eqn:stepIS}
#     \hat{\eta}^{\pi}_{StepDR} = \frac{1}{n} \sum_{i=1}^n  \widehat{V}(S_{i,0}) + 
#     \frac{1}{n} \sum_{i=1}^n \sum_{t=0}^{T - 1} \rho^i_t  \gamma^t \Big[
#     R_{i,t} - \widehat{Q}(A_{i,t}, S_{i,t}) + \gamma \widehat{V}(S_{i,t + 1})
#     \Big]. 
# \end{align*}
# The self-normalized version can be similarly constructed \citep{thomas2016data}. 
# 
# Besides directly applying the DR technique to the value estimator, we can utilize the recursive form 
# to debias the Q- or V-function recursively, which shares a similar idea with our proposal in Section 
# \ref{sec:our_method}. 
# For example, \citet{jiang2016doubly} considered the following estimator. 
# Let $\widehat{V}_{DR}^T = 0$. 
# For $t = T - 1, \dots, 0$, we recursively define 
# \begin{equation*}
#     \widehat{V}_{DR}^t = \frac{1}{n} \sum_{i=1}^n \Big\{ \widehat{V}(S_{i,t}) + \rho^i_t \big[R_{i,t} + \gamma \widehat{V}_{DR}^{t+1}(S_{i,t + 1}) - \widehat{Q}(A_{i,t}, S_{i,t})
#     \big] \Big\}. 
# \end{equation*}
# The final value estimator is then defined as $\widehat{V}_{DR}^0$. 
# 
# The name, doubly robust, reflects the fact that the DR estimators are typically consistent as long as one of the two components is consistent, and hence the estimator is doubly robust to model mis-specifications. 
# Besides, a DR estimator typically has lower (or comparable) bias and variance than its components, in the asymptotic sense.  
# However, similar with the standard IS methods, standard DR estimators also rely on per-step importance ratios  and hence will suffer from huge variance when the horizon is long. 
# We will introduce a state-of-the-art DR estimator in Section \ref{sec:DRL}, which builds on a novel technique to be introduced in Section \ref{sec:curse_horizon} to avoid the curse of horizon. 
# % In Section \ref{sec:curse_horizon}, we will introduce a  technique that helps avoid this issue. 
# 
# % The advantages and limitations of the Double Robust methods should be discussed here. 
# % farajtabar2018more
# 
# 
# 
# 
# \subsection{Double reinforcement learning}\label{sec:DRL}
# % Inspired by the line of research on stationary distribution-based OPE, 
# The superior performance of the stationary distribution-based approach gains growing interest, and a few extensions have been proposed in the literature. 
# For example, 
# \citet{tang2019doubly} designed a DR version, and \citet{uehara2019minimax} proposed to learn a single nuisance function $\widetilde{\xi}^{\pi}(s,a) \equiv \widetilde{\omega}^{\pi}(s) [\pi(a|s) / b(a|s)]$ instead of learning $\widetilde{\omega}^{\pi}(s)$ and $b$ separately. 
# 
# 
# In particular, 
# following this line of research, 
# % following the line of research on stationary distribution-based OPE,  
# \citet{kallus2019efficiently} recently proposed a state-of-the-art method named double reinforcement learning (DRL) that achieves the semiparametric efficiency bound for OPE. 
# DRL is a doubly robust method and our proposal in Section \ref{sec:our_method} is built upon DRL to achieve statistical efficiency. 
# 
# To begin with, we first define the \textit{marginalized density ratio} under the target policy $\pi$ as  
# \vspace{-0.1cm}
# \begin{eqnarray}\label{eqn:omega}
# % =\frac{\pi(a|s)p_{\gamma,\pi}(s)}{b(a|s)p_b(s)}
# 	\omega^{\pi}(a,s)=\frac{(1-\gamma)\sum_{t=0}^{+\infty} \gamma^{t} p_t^{\pi}(a,s)}{p_b(a, s)}, 
# \end{eqnarray}
# where $p_t^{\pi}(a, s)$ denotes the probability of $\{S_t = s, A_t = a\}$ following policy $\pi$ with  $S_{0}\sim \mathbb{G}$. 
# Recall that $p_b(s, a)$ is the stationary density function of the state-action pair under the policy $b$. 
# It is worthy to mention that the denominator of \eqref{eqn:omega} is slightly different from the one in \eqref{eqn:breaking}, in that it is not in a discounted form. 
# This modification avoids throwing away samples with a geometric probability and is  essential to achieve the semiparametric efficiency bound. 
# % $d^{b}(s)b(a|s)$
# 
# 
# % \textbf{The Marginalized Density Ratio.} We next discuss the method for learning $\omega^{\pi}$. In our implementation, we employ the method of \citet{uehara2019minimax}. %Meanwhile, other methods  such as \citet{kallus2019efficiently}, are equally applicable. 
# One critical question is how to estimate the nuisance function $\omega^{\pi}$. 
# The following observation forms the basis: $\omega^{\pi}$ is the only function that satisfies the equation $\Mean L(\omega^{\pi},f)=0$ for any function $f$, where $L(\omega^{\pi},f)$ equals \vspace{-0.2cm}
# \begin{eqnarray}\label{eqn_omega}
# \begin{split}
# 	\Big[\Mean_{a \sim \pi(\cdot|S_{t+1})} \{\omega^{\pi}(A_{t},S_{t})
# 	(\gamma f(a, S_{t+1})- f(A_{t},S_{t}) ) \}
# 	+ (1-\gamma) \Mean_{s \sim \mathbb{G}, a \sim \pi(\cdot|s)} f(a, s). 
# \end{split}
# \end{eqnarray} 
# As such, $\omega^{\pi}$ can be learned by solving the following mini-max problem, \vspace{-0.1cm}
# \begin{eqnarray}\label{eqn:solveL}
# \argmin_{\omega\in \Omega} \sup_{f\in \mathcal{F}} \{\Mean L(\omega, f)\}^2, 
# \end{eqnarray}
# for some function classes $\Omega$ and $\mathcal{F}$. 
# To simplify the calculation, we can choose $\mathcal{F}$ to be a reproducing kernel Hilbert space (RKHS). 
# This yields a closed-form expression for $\sup_{f\in \mathcal{F}} \{\Mean L(\omega,f)\}^2$, for any $\omega$. Consequently, $\omega^{\pi}$ can be learned by solving the outer minimization via optimization methods such as stochastic gradient descent, 
# with the expectation in \eqref{eqn:solveL} approximated by the sample mean. 
# $\widetilde{\omega}^{\pi}(s)$ in \eqref{eqn:breaking} can be learned in a similar manner. 
# % To save space, we defer the details to Appendix \ref{secomega} in the supplementary article. 
# 
# 
# 
# Let $\widehat{Q}$ and $\widehat{\omega}$ be some estimates of  $Q^{\pi}$ and $\omega^{\pi}$,  respectively. 
# DRL first constructs the following estimator for every $(i,t)$ in a doubly robust manner: 
# \vspace{-0.1cm}
# \begin{eqnarray}\label{term}
# \begin{split}
# 	\psi_{i,t}
# 	\equiv
# 	\frac{1}{1-\gamma}\widehat{\omega}(A_{i,t},S_{i,t})\{R_{i,t} 
# 	-\widehat{Q}(A_{i,t},S_{i,t})+
# 	\gamma 
# 	\Mean_{a \sim \pi(\cdot| S_{i,t+1})}\widehat{Q}(a, S_{i,t+1})\}
# 	+ \Mean_{s \sim \mathbb{G}, a \sim \pi(\cdot| s)}\widehat{Q}(a, s). 
# \end{split}	
# \end{eqnarray}
# The resulting value estimator is then given by
# \vspace{-0.1cm}
# \begin{eqnarray*}
# 	\widehat{\eta}_{\tiny{\textrm{DRL}}}=\frac{1}{nT}\sum_{i=1}^n\sum_{t=0}^{T-1} \psi_{i,t}.
# \end{eqnarray*}
# One can show that 
# $\widehat{\eta}_{\tiny{\textrm{DRL}}}$ is consistent when either $\widehat{Q}$ or $\widehat{\omega}$ is consistent, and hence is doubly robust. 
# % . This is referred to as the doubly-robustness property. 
# In addition, %informally speaking, 
# when both $\widehat{Q}$ and $\widehat{\omega}$ converge at a rate faster than $(nT)^{-1/4}$, %it follows from Theorems 9-11 of \citet{kallus2019efficiently} that
# $\sqrt{nT} (\widehat{\eta}_{\tiny{\textrm{DRL}}} - \eta^{\pi})$ converges weakly to a normal distribution with mean zero and variance% $\sigma^2$ as 
# \begin{eqnarray}\label{lower_bound}
#     % \sigma^2 = 
#     \frac{1}{(1-\gamma)^2}\Mean \left[ 
#     \omega^{\pi}(A, S) \{R + \gamma V^{\pi}(S') -  Q^{\pi}(A,S)\}
#     \right]^2,
# \end{eqnarray}
# where the expectation is over tuples following 
# % tuple $(S,A,R,S')$ follows 
# the stationary distribution of the process $\{(S_t,A_t,R_t,S_{t+1})\}_{t\ge 0}$, generated by $b$. 
# See Theorem 11 of \citet{kallus2019efficiently} for a formal proof. 
# % More motivations about DRL will be discussed in Section \ref{sec:our_method}. 
# 
# Moreover, \eqref{lower_bound} is proven to be the \textit{semiparametric efficiency bound} for  infinite-horizon OPE \citep{kallus2019efficiently}. 
# Informally speaking, a semiparametric efficiency bound can be viewed as the nonparametric extension  of the Cramer–Rao lower bound in parametric models \cite{bickel1993efficient}. It provides a lower bound of the asymptotic variance among all regular estimators \cite{van2000asymptotic}. 
# %Refer to \citet{kallus2020double} for a more rigorous definition. 
# Many other OPE methods such as \citet{liu2018breaking}, are statistically inefficient in that their variance are strictly larger than this bound. 
# %\cite{kallus2020double}, and 
# 

# ## Demo [TODO]

# ## References
# 1. Precup D. Eligibility traces for off-policy policy evaluation[J]. Computer Science Department Faculty Publication Series, 2000: 80.
# 2. Thomas P S. Safe reinforcement learning[J]. 2015.
# 3. Jiang N, Li L. Doubly robust off-policy value evaluation for reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2016: 652-661.
# 4. Liu Q, Li L, Tang Z, et al. Breaking the curse of horizon: Infinite-horizon off-policy estimation[J]. Advances in Neural Information Processing Systems, 2018, 31.

# ## Note
# 1. When the behaviour policy is unknown, we can estimate it from data by regarding the task as a classification problem and using methods such as logistic regression. 
# 2. We note that, in principle, IS-based methods (and doubly robust methods to be reviewed in the next section) only apply to the finite-horizon setting, where the  trajectory is truncated at a finite time step $T$. 
# The estimand is 
# $\mathbb{E}^{\pi}_{s \sim \mathbb{G}} (\sum_{t=0}^{T-1} \gamma^t R_{t}|S_{0}=s)$ instead of 
# $\mathbb{E}^{\pi}_{s \sim \mathbb{G}} (\sum_{t=0}^{+\infty} \gamma^t R_{t}|S_{0}=s)$. 
# However, when $T$ is relatively large and $\gamma$ is not quite close to $1$, the difference between $\sum_{t=0}^{T-1} \gamma^t$ and $\sum_{t=0}^{\infty} \gamma^t$ is negligible and is usually ignored, and they are still commonly used as baselines. 
# 3. We note that (SA) is not a strong assumption. Recall that $\{S_{i,t}\}_{t \ge 0}$ is generated by following the stationary policy $b$. (SA) is automatically  satisfied when the initial distribution equals the stationary distribution. Besides, When the MDP is a Harris ergodic chain , the process will eventually mix well and we can replace the stationary distribution with its limiting assumption and the following discussions will continue to hold. 

# In[ ]:




