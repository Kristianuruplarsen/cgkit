""" Components of a graph
"""


class Edge(object):
    """ Graph edge with bias and weight for the mathematical relation.

    Args:
        structure(GraphStructure): the graphstructure in which the edge lives.
        i: in-edge.
        o: out-edge.

    Attributes:
        i: in-edge.
        o: out-edge.
        bias: bias ("intercept")
        weight: weight ("parameter")
    """

    def __init__(self, structure, i, o):
        self.i = i 
        self.o = o 

        self.bias = structure.bias_space()
        self.weight = structure.weight_space()

    def parameters(self):
        """ Return the stored parameters in a tuple.

        Returns:
            tuple: a (bias, weight) tuple for the edge.
        """
        return self.bias, self.weight

    def alter_bias(self, delta):
        """ Change the bias stored in the edge.

        Args:
            delta(float): the amount by which the bias is changed.
        """
        self.bias += delta

    def alter_weight(self, delta):
        """ Change the weight stored in the edge.

        Args:
            delta(float): the amount by which the weight is changed.
        """
        self.weight += delta

