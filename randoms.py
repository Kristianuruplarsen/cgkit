
# Preliminary work on a panel structure with potential
# drift in parameters over time.

#%%
import numpy as np 
import pandas as pd 

from cgkit import CausalGraph


C = CausalGraph(4,
                n_dummies = 1,
                density =0.8,
                seed = 5,
                parameter_space = lambda: np.random.randint(-10,11)
                )


C.draw_graph()

#%%
class CausalPanelGraph(CausalGraph):
    
    def __init__(self,
                n_continuous,
                panel_levels = 1,
                **kwargs):
        """ Causal Graph which implements a panel structure.
        """        
        self.levels = panel_levels 

        if not isinstance(panel_levels, int) or panel_levels < 1:
            raise ValueError("Panel level must be a positive integer")

        super().__init__(n_continuous = n_continuous,
                                          **kwargs) 
        self.yield_cross_section = self.yield_dataset 


    def yield_dataset(self, nobs):
        """ Yield a panel structure dataset over some fixed amount
        of levels. 
        """
        for t in range(self.levels):

            (Xtemp, ytemp), dftemp = self.yield_cross_section(nobs = nobs)

            # Set id for panel period and individual 
            dftemp['t'] = t
            dftemp['i'] = dftemp.index

            try:
                df = pd.concat([df, dftemp]).reset_index(drop = True)
            except NameError:
                df = pd.DataFrame(dftemp)

        self.X = df.drop('Y', axis = 1).values 
        self.y = df['y'].values
        self.df = df

        return (X,y), df

    def parameter_drift(self):
        pass

#%%
C = CausalPanelGraph(4, panel_levels = 1, n_dummies = 1, seed = 5, density = 0.8,
                    parameter_space = lambda: np.random.randint(-10,10)
                    )
C.draw_graph()

#%%
(X,y), data = C.yield_dataset(100)

data.head()

#%%
