# Causal Graphs and Regression
Simulating random causal graphs for experimenting with various regression techniques. This is not finished, so it might take some tinkering to get working.

## Usage

the cgraph class simulates random causal graphs
```python
from data.causalgraph import cgraph


C = cgraph(4, density =1)
C.draw_graph()
```
will yield a graph like

<p align="center">
<img src="figs/examplegraph.png">
</p>

Parameters governing the relation between the variables are stored in `C.parameters` as a dictionary with one key for each variable. Within each key is then a dictionary of `{ancestor:parameter}` pairs

```
{
 0: {},
 1: {2: 5, 0: 1},
 2: {0: 1},
 3: {1: 2},
 'Y': {2: 2, 1: 4}
}
```
We can simulate a row-column dataset with

```python
C.yield_dataset(nobs = 1000)
```
which will give a numpy array with data.
