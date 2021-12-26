<h1><img src="logo.png" width="100px" align="left" style="margin-right: 10px;"> Causal Decision Making: An Tutorial for Personalized Decision-Making</h1>

This repository contains the source code of [Causal Decision Making](http://causaldm.com/), an online tutorial with the goal of providing practitioners a systematic review, map, and handbook for finding the appropriate solutions for their problems. 
The tutorial surveys and discusses classic methods and recent advanced in the the off-policy learning literature, and complements the [CausalDM](https://github.com/CausalDM/CausalDM) Python package where open-source implementations are provided under a unified API. 

## How to contribute
### Publish a new version
1. switch to the main branch
2. `jupyter-book build Causal-Decision-Making` (the static html should then be visiable locally)
4. `cd Causal-Decision-Making`
3. `ghp-import -n -p -f _build/html`
4. push both the main branch and the gh-pages branch
5. One command line: `cd Documents; jupyter-book build Causal-Decision-Making; cd Causal-Decision-Making; ghp-import -n -p -f _build/html`
