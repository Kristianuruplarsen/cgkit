
""" Randomly generate directed graphs to use as causal graph
"""

# TODO: 
#   . Refactor a bit more
#   . Allow for dummy outcome 
#   . Allow for categorical variables
#   . Add biases to variables 
# WORK IN PROGRESS:
#   . include biases

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from .graphUtils import GraphUtils
#from .dataYielder import DataYielder



class DataYielder():
    
    def __init__(self,
                nobs,
                weights,
                noise_space
                ):

        self.nobs = nobs
        self.weights = weights
        self.noise_space = noise_space

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
            random_data += (self.weights[param_id][var_id]) * X[var_id] 

        if param_id[0] == 'd':
            binary_data = np.where( random_data > np.mean(random_data), 1, 0)
            return binary_data
        return random_data
        



class CausalGraph(GraphUtils):
    """Simulate a full causal graph.
    """
    def __init__(self,
                n_continuous,
                n_dummies = 0,
                density = 1,
                edges = "random",
                seed = None,
                parameter_space = lambda:  np.random.randint(1,6),
                noise_space = lambda nobs: np.random.normal(size = nobs),
                ):
        """ 
        Simulate a causal graph

        Parameters
        ----------
            n_continuous: int 
                Number of continous X variables
            n_dummies: int 
                Number of binary X variables
            density: float 
                Graph connectedness (between 0,1)
            edges: list 
                Manually constructed edges or default="random" for
                randomly generated edges.
            seed: int
                Seed
            parameter_space: func 
                Lambda function that returns parameter values
                default=np.random.randint(1,6)
            noise_space: func 
                What distribution should we draw from for the
                undetermined component of variables? default is a std. gaussian.
                Must be a function of the number of observations!

        Attributes
        ----------
        A lot
        """
        self._check_inputs(continuous = n_continuous,
                           dummies = n_dummies,
                           density = density
                          )

        self.seed = seed
        self.n_continuous = int(n_continuous)
        self.n_dummies = int(n_dummies)

        self.nvars = self.n_continuous + self.n_dummies
        self.density = density

        if self.seed is not None:
            np.random.seed(self.seed)

        self.parameter_space = parameter_space
        self.noise_space = noise_space

        self.ncon = self._calculate_n_connections(self.nvars, self.density)

        self.continuous_nodes = [f'c{i}' for i in range(self.n_continuous)]
        self.binary_nodes = [f'd{i}' for i in range(self.n_dummies)]

        self.nodelist = self.continuous_nodes + \
                        self.binary_nodes + \
                        ['Y']

        # Set up graph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodelist)

        if edges == "random":
            self.edges = self._random_edges()
            for e in self.edges:
                self.G.add_edge(*e)
        else:
            self.edges = edges
            for e in self.edges:
                self.G.add_edge(*e)

        self.weights = self._weights()
        self.biases = None

    def _random_edges(self):
        """ 
        Random edges.

        Randomly constructs causal edges between variables. 
        For internal use.
        """
        e = list()

        while len(e) < self.ncon:

            bidx = np.random.choice(len(self.nodelist))
            begin = self.nodelist[bidx]

            eidx = np.random.choice(len(self.nodelist))
            end = self.nodelist[eidx]

            if not self._new_node_ok(begin, end, e):
                continue

            if not self._graph_remains_DAG(begin, end, e):
                continue

            e.append((begin, end))

        return e

    def _biases(self):
        return None

    def _weights(self):
        """ 
        Draw weights for linear relation between variables. 
        Mainly for internal use.
        """
        return {n: {k[0]: self.parameter_space() for k in self.G.in_edges(n)} for n in self.G.nodes}


    def yield_dataset(self, nobs, **kwargs):
        """ 
        Simulate rows and columns from the causal graph by repeatedly
        filling the columns for which we know all ancestors until the entire
        dataset is complete.

        Parameters
        ----------
            nobs: int 
                Number of observations.

            **kwargs: 
                Other arguments to DataYielder()
        Returns
        -------
            data: tuple
                A tuple of X,y as numpy arrays and a full pandas dataframe df,
                i.e. (X,y),df
        """
        done = {k: False for k in self.weights.keys()}

        dy = DataYielder(nobs = nobs,
                        weights = self.weights,
                        noise_space = self.noise_space,
                        **kwargs)

        X = pd.DataFrame( np.zeros(shape = (nobs, len(self.G.nodes))) )
        X.columns = self.nodelist

        while not all(done.values()):
            for par in [p for p in done if not done[p]]:

                # Independent variables
                if len(self.weights[par]) == 0 and not done[par]:
                    X[par] = dy._make_independent(param_id = par)
                    done[par] = True

                # If all ancestors are made
                if all({k: done[k] for k in self.weights[par].keys()}.values()) and not done[par]:
                    X[par] = dy._make_dependent(X, par)
                    done[par] = True

        # Set attributes in self
        self.X = X.drop('Y', axis = 1).values
        self.y =  X['Y'].values
        self.df = X

        # Return relevant attributes
        data = (self.X, self.y), self.df
        return data



    def wintervene(self, effect, cause, parameter, unsafe = False):
        """ 
        Intervene on a specific parameter.
        
        Replaces the parameter b in the equation effect = b*cause + ... with 
        something new.
        
        Parameters
        ----------
            effect: int/'Y' 
                The variable whose equation to intervening on.
            cause: int 
                The variable to get a new parameter in the equation for
                effect.
            parameter: float 
                New parameter value.
            unsafe: bool 
                If unsafe you can create weights for relations that
                do not exist. Default False.
        """
        if not isinstance(parameter,(float, int)):
            raise ValueError("parameter must be a number.")

        if not unsafe:
            try:
                self.weights[effect][cause] = parameter
                return self
            except:
                raise KeyError("Creating new connections can jeopardize DAG structure")
        else:
            self.weights[effect][cause] = parameter
            return self


    def bintervene(self, cause, transformation):
        raise NotImplementedError("cannot intervene on values yet.")


    def draw_graph(self):
        """ Plot the simulated graph.
        """
        p = nx.circular_layout(self.G)

        if len(self.continuous_nodes) > 0:
            nodes1 = nx.draw_networkx_nodes(self.G,
                                nodelist = self.continuous_nodes,
                                pos = p,
                                node_color = 'royalblue',
                                alpha = 1)
            nodes1.set_edgecolor('black')

        if len(self.binary_nodes) > 0:
            nodes2 = nx.draw_networkx_nodes(self.G,
                                nodelist = self.binary_nodes,
                                pos = p,
                                node_color = 'white',
                                alpha = 1)
            nodes2.set_edgecolor('black')

        nodes3 = nx.draw_networkx_nodes(self.G,
                               nodelist = ['Y'],
                               pos = p,
                               node_color = 'red',
                               alpha = 0.8)
        nodes3.set_edgecolor('black')

        nx.draw_networkx_edges(self.G,
                              pos = p,
                              width = 1.0)

        nx.draw_networkx_labels(self.G, pos = p)
        plt.axis('off')
        return None


