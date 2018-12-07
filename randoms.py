
# Preliminary work on a panel structure with potential
# drift in parameters over time.

#%%
import numpy as np 
import pandas as pd 

from cgkit import CausalGraph, CausalPanelGraph

#%%

C = CausalPanelGraph(
                panel_levels = 5,
                n_continuous = 4,
                n_dummies = 1,
                density =0.8,
                seed = 5,
                parameter_space = lambda: np.random.randint(-10,11)
                )


C.draw_graph()

#%%
(X,y), df = C.yield_panel_dataset(100)

#%%
