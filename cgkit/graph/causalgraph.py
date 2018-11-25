
""" Randomly generate directed graphs to use as causal graph
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

class CausalGraph:
    """Simulate a full causal graph.
    """
    def __init__(self,
                noX,
                density = 1,
                edges = "random",
                seed = None,
                parameter_space = lambda:  np.random.randint(1,6),
                noise_space = lambda nobs: np.random.normal(size = nobs),
                ):
        """ Simulate a causal graph

        Args:
            noX (int): Number of X variables
            density (float): graph connectedness (between 0,1)
            edges (list): manually constructed edges or default="random" for
                randomly generated edges.
            seed (int): seed
            parameter_space (func): lambda function that returns parameter values
                default=np.random.randint(1,6)
            noise_space (func): What distribution should we draw from for the
            undetermined component of variables? default is a std. gaussian.
                Must be a function of the number of observations!
        """

        self.seed = seed
        self.nvars = int(noX)

        if self.seed is not None:
            np.random.seed(self.seed)

        self.parameter_space = parameter_space
        self.noise_space = noise_space

        if self.nvars <= 0:
            raise ValueError("input noX must be a positive integer")

        if not 0 < density <= 1:
            raise ValueError("Density must be in the interval ]0,1]")

        self.ncon = np.ceil(density* ((self.nvars - 1)*(self.nvars))/2)
        self.nodelist = [i for i in range(self.nvars)] + ['Y']
        self.G = nx.DiGraph()

        # Add X-variables 0-nvars and Y
        self.G.add_nodes_from(self.nodelist)

        if edges == "random":
            self.edges = self._random_edges()
            for e in self.edges:
                self.G.add_edge(*e)
        else:
            self.edges = edges
            for e in self.edges:
                self.G.add_edge(*e)

        self.parameters = self._parameters()


    def _random_edges(self):
        """ Randomly constructs causal edges between variables. Internal
        """
        e = list()

        while len(e) < self.ncon:

            bidx = np.random.choice(len(self.nodelist))
            begin = self.nodelist[bidx]

            eidx = np.random.choice(len(self.nodelist))
            end = self.nodelist[eidx]

            if begin == 'Y' \
            or begin == end \
            or (begin, end) in e \
            or (end, begin) in e:
                continue

            # Ensures result is a DAG
            if len(list(nx.simple_cycles(nx.DiGraph(e + [(begin, end)])))) > 0:
                continue

            e.append((begin, end))

        return e


    def draw_graph(self):
        """ Plot the simulated graph.
        """
        p = nx.circular_layout(self.G)
        nodes1 = nx.draw_networkx_nodes(self.G,
                              nodelist = [i for i in self.G.nodes if not i == 'Y'],
                              pos = p,
                              node_color = 'skyblue',
                              alpha = 1)

        nodes2 = nx.draw_networkx_nodes(self.G,
                               nodelist = ['Y'],
                               pos = p,
                               node_color = 'red',
                               alpha = 0.8)

        nodes1.set_edgecolor('black')
        nodes2.set_edgecolor('black')

        nx.draw_networkx_edges(self.G,
                              pos = p,
                              width = 1.0)

        nx.draw_networkx_labels(self.G, pos = p)
        plt.axis('off')
        return None

    def _parameters(self):
        """ Draw parameters for linear relation between variables. Internal
        """
        return {n: {k[0]: self.parameter_space() for k in self.G.in_edges(n)} for n in self.G.nodes}


    def yield_dataset(self, nobs):
        """ Simulate rows and columns from the causal graph by repeatedly
        filling the columns for which we know all ancestors until the entire
        dataset is complete.

        Args:
            nobs (int): number of observations.
        """
        done = {k: False for k in self.parameters.keys()}
        X = np.zeros(shape = (nobs, len(self.G.nodes)))

        while not all(done.values()):
            for par in [p for p in done if not done[p]]:
                # Independent variables
                if len(self.parameters[par]) == 0 and not done[par]:
                    X[:,par] = self.noise_space(nobs)
                    done[par] = True

                # If all ancestors are made
                if all({k: done[k] for k in self.parameters[par].keys()}.values()) and not done[par]:

                    for var in self.parameters[par]:
                        if par == 'Y':
                            X[:,-1] += self.parameters[par][var] * X[:,var] + self.noise_space(nobs)
                            done[par] = True
                        else:
                            X[:,par] += self.parameters[par][var]*X[:,var] + self.noise_space(nobs)
                            done[par] = True
        # Set attributes in self
        self.X = X[:,:-1]
        self.y =  X[:,-1]
        self.df = pd.DataFrame(X)
        self.df.columns = self.nodelist

        # Return relevant attributes
        return (self.X, self.y), self.df


    def pintervene(self, effect, cause, parameter, unsafe = False):
        """ Intervene on a specific parameter.
        Replaces the parameter b in the equation
        effect = b*cause + ... with something new.
        Args:
            effect (int/'Y'): the variable whose equation to intervening on.
            cause (int): The variable to get a new parameter in the equation for
                effect.
            parameter (float): new parameter value.
            unsafe (bool): if unsafe you can create parameters for relations that
                do not exist. Default False.
        """
        if not isinstance(parameter,(float, int)):
            raise ValueError("parameter must be a number.")

        if not unsafe:
            try:
                self.parameters[effect][cause]
            except:
                raise KeyError("Creating new connections can jeopardize DAG structure")

        self.parameters[effect][cause] = parameter
        return self


    def vintervene(self, cause, transformation):
        pass
