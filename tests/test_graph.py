#%%
from cgkit import (CausalGraph,
                   DataYielder,
                   GraphUtils)

import numpy as np
#%%
def test_linear_nonlinear():
    C = CausalGraph(5)
    f = C._yield_linear_nonlinear_effects(C._FUNCS, 0.5)
    f(4)
    f(-4)

test_linear_nonlinear()

#%%
def test_yielder_kwargs():
    C = CausalGraph(5, seed = 1)
    C.yield_dataset(10, p_linear_function = 1)

test_yielder_kwargs()

#%%
def test_graph():
    C = CausalGraph(n_continuous=5)
    return True 
assert test_graph() == True

#%%
def test_wintervene():
    C = CausalGraph(4,
                n_dummies = 1,
                density =0.8,
                seed = 5,
                parameter_space = lambda: np.random.randint(-10,11)
                )
    C.wintervene(effect = 'c1', cause = 'c0', parameter = 5)

    if C.weights['c1']['c0'] == 5:
        return True 
    return False

assert test_wintervene() == True

#%%

def test_node_ok():
    C = CausalGraph(5, n_dummies=1, density=0.1, seed = 5)

    assert C._new_node_ok(1,1, C.edges) == False 
    assert C._new_node_ok('Y',1, C.edges) == False
    assert C._new_node_ok('c2', 'Y', C.edges) == True 

    return True 
assert test_node_ok() == True

#%%

def test_value_checks():
    C = CausalGraph(5, n_dummies=1, density=0.1, seed = 5)
    f = list()
    try:
        C = CausalGraph(0, n_dummies=0, density=0.1, seed = 5)
    except:
        f.append(1)
    try:
        C = CausalGraph(5, n_dummies=-1, density=0.1, seed = 5) 
    except:
        f.append(2)

    return f

assert test_value_checks() == [1,2]
#%%    

def test_draw():
    C = CausalGraph(5, n_dummies=1, density=0.1, seed = 5)
    C.draw_graph()

    C = CausalGraph(5, n_dummies=0, density=0.1, seed = 5)
    C.draw_graph()

    C = CausalGraph(0, n_dummies=5, density=0.1, seed = 5)
    C.draw_graph()



test_draw()

#%%

def test_nonlinear():
    C = CausalGraph(5, n_dummies= 1, seed = 5)
    n = C.nonlinearity_links
    assert len(n) == len(C.G.nodes)

test_nonlinear()

#%%
def test_nonlinear_2():
    C = CausalGraph(5, n_dummies= 1, seed = 5)
    assert str(type(C.nonlinearity_links['c0']['c1'])) == "<class 'function'>"


test_nonlinear_2()

#%%
