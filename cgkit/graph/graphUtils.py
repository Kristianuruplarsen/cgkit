""" 
Utility functions for the CausalGraph class
"""
import numpy as np 
import networkx as nx

class GraphUtils:

    @staticmethod
    def _new_node_ok(begin, end, edges):
        """
        Check a node can be added to the graph. For internal use.

        Parameters
        ----------
        begin: node
            Beginning node 
        end: node
            End node 
        edges: list
            List of already added edges
        
        Returns
        -------
        boolean (ok/not ok)
        """
        if begin == 'Y' \
        or begin == end \
        or (begin, end) in edges \
        or (end, begin) in edges:
            return False
        return True


    @staticmethod
    def _graph_remains_DAG(begin, end, edges):
        """ 
        Will adding begin, end break the DAG structure?
        """
        if len(list(nx.simple_cycles(nx.DiGraph(edges + [(begin, end)])))) > 0:
            return False
        return True


    @staticmethod
    def _check_inputs(continuous, dummies, density, p_linear):
        """ 
        Check that inputs to CausalGraph are ok.
        """
        if not continuous >= 0 and isinstance(continuous, int):
            raise ValueError("Input n_continuous must be a positive integer.")

        if not dummies >= 0 and isinstance(dummies, int):
            raise ValueError("Input n_dummies must be a positive integer.")

        if not continuous + dummies > 0:
            raise ValueError("Must have more than 0 X variables.")
            
        if not 0 < density <= 1:
            raise ValueError("Density must be in the interval ]0,1].")

        if not 0 <= p_linear <= 1:
            raise ValueError("p_linear must be in the interval [0,1].")

    @staticmethod
    def _calculate_n_connections(n_vars, density):
        """
        Number of graph edges calculated from the number of variables and 
        the density.
        """
        return np.ceil(density * ( (n_vars - 1)*(n_vars)) / 2 )


    @staticmethod
    def _yield_linear_nonlinear_effects(functions_list, p_linear, cause):
        """
        Randomly yield a linear or nonlinear function to apply to a variable.
        """
        # Always linear effect from dummies.
        if cause[0] == 'd':
            idx = 0

        else:
            idx = np.random.choice(len(functions_list), p = [p_linear] +\
                [(1 - p_linear) / (len(functions_list) - 1) for _ in functions_list][1:]
                )            
        return functions_list[idx]
        
