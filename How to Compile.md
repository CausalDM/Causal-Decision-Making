## How to compile

This book is powered by a package call [Jupyter Book](https://jupyterbook.org/intro.html). 
Refer to its documents for more details. 

To compile, we need to first install this package

```
pip install -U jupyter-book == 0.15.1
```

### Step 1: Compile a new version
1. in Github Desktop, switch to the main branch
2. in your terminal, go to the parent folder of `Causal-Decision-Making`
3. run `jupyter-book build Causal-Decision-Making`
4. The static html should then be visiable locally at `Causal-Decision-Making/_build/html/index.html`
5. Commit & push this version (main branch)

If you only want to share this version internally instead of publishing it online, the steps above are enough. 
Otherwise, continue with the steps in the next section. 

### Step 2 (optional): Publish a new version
4. `cd Causal-Decision-Making`
5. run `ghp-import -n -p -f _build/html` at the `main` branch. Ignores the password part.  
6. push the `gh-pages` branch

### Potential Error Messages when Running ghp-import
1. File not found error: you may not have Git installed. Try installing/reinstalling the Git 
2. Fail identification: the credential information saved in your local credential manager needs to be updated. You can generate personal access token at https://github.com/settings/tokens
3. The default website link is https://causaldm.github.io/Causal-Decision-Making/
  - If we want to change the link to the 'causaldm.com': manually add a 'CNAME' file to the `gh-pages` branch. Then, within the 'CNAME' file, type `causaldm.com`.


One command line for the two steps: 
```
cd Documents; jupyter-book build Causal-Decision-Making; cd Causal-Decision-Making; ghp-import -n -p -f _build/html
```
