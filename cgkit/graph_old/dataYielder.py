"""
Class handling background work when yielding dataset from a causal graph
"""

import numpy as np 


class DataYielder():

    def __init__(self,
                nobs,
                weights,
                noise_space,
                functions,
                seed,
                p_linear_function = 0.5
                ):

        self.nobs = nobs
        self.weights = weights
        self.noise_space = noise_space
        self.link_functions = functions
        self.seed = seed
        self.plinear = p_linear_function 

        if self.seed is not None:
            np.random.seed(self.seed)


    def _make_independent(self, param_id):
        """
        Random normal variable with no predictors.
        """
        random_data = self.noise_space(self.nobs)

        if param_id[0] == 'd':
            binary_data = np.where( random_data > np.mean(random_data), 1, 0)
            return binary_data

        return random_data


    def _make_dependent(self, X, param_id):
        """
        Random normal variable plus any causal effects from other 
        variables in the graph.
        """
        random_data = self.noise_space(self.nobs)

        for var_id in self.weights[param_id]:
            functype, func = self.link_functions[param_id][var_id]
            random_data += self.weights[param_id][var_id] * func(X[var_id])

        if param_id[0] == 'd':
            binary_data = np.where( random_data > np.mean(random_data), 1, 0)
            return binary_data
        return random_data




