""" Utility functions for working with graphs.
"""

import numpy as np 


def as_balanced_dummy(X):
    """ Convert continous to balanced binary.
    """
    return np.where(X > np.mean(X), 1, 0)


def find_true_values(structure, kind = 'weights'):
    """ Find the true parameter values of a graph structure.

    Args:
        structure(GraphStructure): A graph structure.

    Returns:
        a dictionary of true parameter values.
    """
    true_vals = dict()

    for edge in structure.edges:
        if edge.o[1] == '0':

            if kind == 'weights':
                true_vals[edge.i] = edge.weight
            elif kind == 'biases':
                true_vals[edge.i] = edge.bias
            else:
                raise ValueError("kind must be 'weights' or 'biases'.")

        else:
            if edge.i in true_vals:
                continue
            true_vals[edge.i] = 0

    return true_vals


def find_edge(edges, i, o):
    """ Search through a list of edges and return the one from i to o.

    Args:
        edges(list): a list of `Edge` instances. 
        i(str): in-edge.
        o(str): out-edge

    Returns:
        Edge: The (first) matching edge.

    Raises:
        KeyError: the edge from i to o was not in the edgelist.
    """
    for edge in edges:
        if edge.i == i and edge.o == o:
            return edge 
    raise KeyError("edge not in edgelist")
