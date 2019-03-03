
import numpy as np 
import pandas as pd 
import random 
import networkx as nx 

import matplotlib.pyplot as plt 



def find_edge(edges, i, o):
    for edge in edges:
        if edge.i == i and edge.o == o:
            return edge 
    raise KeyError("edge not in edgelist")


class Edge(object):

    def __init__(self, structure, i, o):
        self.i = i 
        self.o = o 

        self.bias = structure.bias_space()
        self.weight = structure.weight_space()

    def parameters(self):
        return self.bias, self.weight

    def alter_bias(self, delta):
        self.bias += delta

    def alter_weight(self, delta):
        self.weight += delta

    

class GraphStructure(object):
    
    def __init__(self,
                 continous: int,
                 dummies: int,
                 density: float,
                 seed: int = None
                ):

        self.density = density 
        self.seed = seed 

        self.n_continous = continous
        self.n_dummies = dummies
        self.n_variables = continous + dummies
        self.approx_n_edges = self._edges()

        # Hidden things
        self._bias_space = lambda: np.random.uniform(0,1)
        self._weight_space = lambda: np.random.uniform(0,1)
        self.max_degree = 3

        if self.seed is not None:
            np.random.seed(self.seed) 

        self.graph = self._set_graph()
        self.edges = [Edge(self, i, o) for i, o in self.graph.edges()]

    @property 
    def bias_space(self):
        return self._bias_space

    @bias_space.setter
    def bias_space(self, func):
        self._bias_space = func
        self.edges = [Edge(self, i, o) for i, o in self.graph.edges()]

    @property 
    def weight_space(self):
        return self._weight_space

    @weight_space.setter 
    def weight_space(self, func):
        self._weight_space = func
        self.edges = [Edge(self, i, o) for i, o in self.graph.edges()]

    def _reseed(self):
        """ Reseed the class to ensure replicability """
        if self.seed is not None:
            np.random.seed(self.seed) 

    def _edges(self): 
        """ Number of edges calculated from edge density """
        max_edges = sum([n - 1 for n in range(1, self.n_variables + 1)])
        return round(self.density*max_edges)

    # Build the graph
    def _set_graph(self, reseed: bool = True):
        """ Set the graph object, should only be used 
            if the graph structure needs to be recomputed.
        """
        if reseed:
            self._reseed()

        graph = self._graph_make_raw(self.n_variables, self.approx_n_edges)
        graph = self._graph_assign_y(graph)
        graph = self._graph_assign_type(graph)

        return graph

    def _graph_make_raw(self, nodes, edges):
        """ Generate a random DAG with raw labels and no
            other data.
        """
        G = nx.DiGraph()

        for i in range(nodes):
            G.add_node(i)
        while edges > 0:
            a = np.random.randint(0, nodes)
            b=a
            while b==a:
                b = np.random.randint(0, nodes)
            G.add_edge(a,b)
            if nx.is_directed_acyclic_graph(G):
                edges -= 1
            else:
                # we closed a loop!
                G.remove_edge(a,b)
        return G


    def _graph_assign_type(self, graph):
        """ Assign a type (c-continous) or (d-dummy) to
            every variable in the graph.
        """
        def relabeller(x, continous, dummies):
            if x in continous:
                return f'c{x}'
            elif x in dummies:
                return f'd{x}'
            return x

        continous = set(random.sample(graph.nodes, self.n_continous))
        dummies = set(graph.nodes) - continous

        graph = nx.relabel_nodes(graph, lambda x: relabeller(x, continous, dummies))

        return graph


    def _graph_assign_y(self, graph):
        """ Relabel the last node in a DAG to 'y'.
        """
        topo = nx.topological_sort(graph)
        *_, last = topo 
        graph = nx.relabel_nodes(graph, lambda x: 'y' if x == last else x)
        return graph 


    def draw_graph(self):
        """ Plot the simulated graph.
        """
        p = nx.circular_layout(self.graph)

        try:
            nodes1 = nx.draw_networkx_nodes(self.graph,
                                nodelist = [x for x in self.graph.nodes if x[0] == 'c' and x[1] != 'y'],
                                pos = p,
                                node_color = 'royalblue',
                                alpha = 1)
            nodes1.set_edgecolor('black')
        except:
            print("Error plotting continous")
        try:
            nodes2 = nx.draw_networkx_nodes(self.graph,
                                nodelist = [x for x in self.graph.nodes if x[0] == 'd' and x[1] != 'y'],
                                pos = p,
                                node_color = 'white',
                                alpha = 1)
            nodes2.set_edgecolor('black')
        except:
            print("Error plotting dummies")
        nodes3 = nx.draw_networkx_nodes(self.graph,
                               nodelist = [x for x in self.graph.nodes if x[1] == 'y'],
                               pos = p,
                               node_color = 'red',
                               alpha = 0.8)
        nodes3.set_edgecolor('black')

        nx.draw_networkx_edges(self.graph,
                              pos = p,
                              width = 1.0)

        nx.draw_networkx_labels(self.graph, pos = p)
        plt.axis('off')
        return plt


    def yield_data(self,
                   nobs: int,
                   reseed: bool = False
                   ):
        """ Return a np array with data generated from the graph.
        """
        if reseed:
            self._reseed()

        topo = list(nx.topological_sort(self.graph))
        data = np.random.randn(nobs, self.n_variables)
        
        for idx, o in enumerate(topo):
            nodetype = o[0]
            inedges = [x[0] for x in self.graph.in_edges(o)]

            if len(inedges) == 0:
                data[:,idx] = np.random.randn(nobs)

            for i in inedges:
                idx_i = topo.index(i)
                b, w = find_edge(self.edges, i, o).parameters()

                data[:,idx] += b + w*data[:,idx_i]

            if nodetype == 'd':
                data[:,idx] = self._as_dummy(data[:,idx])

        return data


    def _as_dummy(self, X):
        return np.where(X > np.mean(X), 1, 0)

    def columns(self):
        topo = list(nx.topological_sort(self.graph))        
        return topo

class CausalGraph(object):

    def __init__(self, 
                 structure: GraphStructure,
                 T: int,
                 nobs: int,
                 reseed: bool = False
                 ):
        self.structure = structure
        self.T = T
        self.t = 0
        self.nobs = nobs 
        self.reseed = reseed 
        self.cols = self.structure.columns()

        self.data = None

    def step(self):
        if self.t >= self.T:
            raise ValueError("Ran through all steps")

        tdata = self.structure.yield_data(nobs = self.nobs, reseed = self.reseed)

        if self.data is None:
            self.data = pd.DataFrame(tdata)
            self.data.columns = self.cols
            self.data['idx'] = self.t
        else:
            tdata = pd.DataFrame(tdata)
            tdata.columns = self.cols
            tdata['idx'] = self.t
            self.data = self.data.append(tdata, ignore_index = False).reset_index(drop = True)
        self.t += 1

    
    def step_all(self):
        while self.t < self.T:
            self.step()
        return self.data




