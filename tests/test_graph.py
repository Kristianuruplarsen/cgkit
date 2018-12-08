#%%
from cgkit import CausalGraph
#%%

def test_graph():
    C = CausalGraph(n_continuous=5)
    return True 

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


def test_node_ok():
    C = CausalGraph(5, n_dummies=1, density=0.1, seed = 5)

    assert C._new_node_ok(1,1, C.edges) == False 
    assert C._new_node_ok('Y',1, C.edges) == False
    assert C._new_node_ok('c2', 'Y', C.edges) == True 

    return True 


def test_value_checks():
    C = CausalGraph(5, n_dummies=1, density=0.1, seed = 5)
    
    try:
        # Should fail
        C = CausalGraph(0, n_dummies=0, density=0.1, seed = 5)
        C = CausalGraph(5, n_dummies=-1, density=0.1, seed = 5) 
        return True
    except:
        return False
    

def test_draw():
    C = CausalGraph(5, n_dummies=1, density=0.1, seed = 5)
    C.draw_graph()

    C = CausalGraph(5, n_dummies=0, density=0.1, seed = 5)
    C.draw_graph()

    C = CausalGraph(0, n_dummies=5, density=0.1, seed = 5)
    C.draw_graph()


assert test_graph() == True
assert test_node_ok() == True
assert test_wintervene() == True
test_draw()

#%%
