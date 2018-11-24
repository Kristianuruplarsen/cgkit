
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from data.causalgraph import cgraph


C = cgraph(5, density =1)
C.draw_graph()


C.G.edges(data = True)

y = np.zeros(shape = 1000)
X = np.zeros(shape = (1000, C.nvars))
