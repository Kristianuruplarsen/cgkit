
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from data.causalgraph import cgraph


C = cgraph(2, density =10)
C.draw_graph()



2+2

C.edges






class simdata:

    def __init__(self, causalgraph):
        self.G = causalgraph
