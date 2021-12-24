# Causal Decision Making


## How to contribute
### Publish a new version
1. switch to the main branch
2. `jupyter-book build Causal-Decision-Making`
	3. the static html should then be visiable locally.
4. `cd Causal-Decision-Making`
3. `ghp-import -n -p -f _build/html`
4. push both the main branch and the gh-pages branch
5. One command line: `cd Documents; jupyter-book build Causal-Decision-Making; cd Causal-Decision-Making; ghp-import -n -p -f _build/html`