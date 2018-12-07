import numpy as np
import pandas as pd 

from .causalgraph import CausalGraph

# Is this even a nice feature?
# TODO: fixed effects on ID:
# within each i observations should 
# be added some unique constant. 

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

    def yield_panel_dataset(self, nobs):
        """ Yield a panel structure dataset over some fixed amount
        of levels. 
        """
        for t in range(self.levels):

            (Xtemp, ytemp), dftemp = self.yield_dataset(nobs = nobs)

            # Set id for panel period and individual 
            dftemp['t'] = t
            dftemp['i'] = dftemp.index

            try:
                df = pd.concat([df, dftemp]).reset_index(drop = True)
            except NameError:
                df = pd.DataFrame(dftemp)

        self.X = df.drop('Y', axis = 1).values 
        self.y = df['Y'].values
        self.df = df

        return (self.X, self.y), self.df

