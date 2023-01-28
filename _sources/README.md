<h1><img src="logo.png" width="90px" align="left" style="margin-right: 10px;"> Causal Decision Making: a Tutorial for Optimal Decision-Making</h1>



This repository contains the source code of [Causal Decision Making](http://causaldm.com/), an online tutorial with the goal of providing practitioners a systematic review, map, and handbook for finding the appropriate solutions for their problems. 
The tutorial surveys and discusses classic methods and recent advanced in the the off-policy learning literature, and complements the [CausalDM](https://github.com/CausalDM/CausalDM) Python package where open-source implementations are provided under a unified API. 

## Content of every notebook
1. Describe the main ideas, advantages, and appropriate use cases (theoretical guarantee?)
2. Introduce the key formulae (not be heavy in notation) and algorithms (pseudo code)
3. Demo of how to use the package

## How to contribute

This book is powered by a package call [Jupyter Book](https://jupyterbook.org/intro.html). 
Refer to its documents for more details. 
To compile, we need to first install this package

```
pip install -U jupyter-book
```

### Compile a new version
1. switch to the main branch in Github Desktop
2. `jupyter-book build Causal-Decision-Making`
3. The static html should then be visiable locally at `Causal-Decision-Making/_build/html/index.html`
4. Commit & push this version (main branch)

If you only want to share this version internally instead of publishing it online, the steps above are enough. 
Otherwise, continue with the steps in the next section. 

### Publish a new version
4. `cd Causal-Decision-Making`
3. run `ghp-import -n -p -f _build/html` at the `main` branch. Ignores the password part.  
4. push the `gh-pages` branch


One command line for the two steps: 
```
cd Documents; jupyter-book build Causal-Decision-Making; cd Causal-Decision-Making; ghp-import -n -p -f _build/html
```
