# Causal Graphs and Regression
Simulating random causal graphs for experimenting with various regression techniques. This is not finished, so it might take some tinkering to get working.

## Usage

the `CausalGraph` class simulates random causal graphs (these are directed and acyclic, although im not entirely sure if this is required). By default the generator is not seeded, but you can simply set a seed if you want to.
```python
from data.causalgraph import CausalGraph

C = CausalGraph(5,
                density =0.8,
                seed = 1,
                parameter_space = lambda: np.random.randint(-10,11)
                )
C.draw_graph()
```
will yield a graph like

<p align="center">
<img src="figs/examplegraph.png">
</p>

Which shows the simulated causal relation between each variable. Parameters governing the relation between the variables are stored in `C.parameters` as a dictionary with one key for each variable. Within each key is then a dictionary of `{ancestor:parameter}` pairs

```
{0: {4: -6, 2: 5},
 1: {0: -8, 4: -3},
 2: {},
 3: {},
 4: {2: -2, 3: -1},
 'Y': {3: -7, 0: -3}}
```
The parameters are drawn randomly (in this case as integers between -10 and 10). We can simulate a row-column dataset with
```python
C.yield_dataset(nobs = 1000)
```
which will give a numpy array with data which looks like:

<p align="center">
<img src="figs/pairs.png">
</p>

Next we can intervene on one of the parameters in the network using `pintervene`
```python
C.pintervene(effect = 0, cause = 2, parameter = -5)
```
This code changes the parameter in the equation X0 = 5*X2 to a 2 (i.e. X0 = -5*X2). When overlaying the new and old dataset we get a figure like

<p align="center">
<img src="figs/pairs2.png">
</p>
