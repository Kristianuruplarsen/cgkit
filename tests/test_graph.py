#%%
from cgkit import CausalGraph
#%%

def test_graph():
    C = CausalGraph(n_continuous=5)
    return True 


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
    

assert test_graph() == True
assert test_node_ok() == True


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
