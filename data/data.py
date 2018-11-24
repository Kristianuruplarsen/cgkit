
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from data.causalgraph import cgraph


C = cgraph(5, density =1)
C.draw_graph()


np.random.randint(-5,6, size = 3)
C.G.in_edges('Y')

[
('Y': (1,3,5)),
...
]

C.G.in_edges('Y')
# Build parameters

params = {n: {k[0]: np.random.randint(-5,6) for k in C.G.in_edges(n)} for n in C.G.nodes}



params
