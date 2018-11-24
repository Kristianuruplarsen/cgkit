
import numpy as np
import networkx as nx
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data.causalgraph import cgraph


C = cgraph(4, density =1)
C.draw_graph()
plt.savefig("figs/examplegraph.png")

C.parameters

C.yield_dataset(1000)

lol = pd.DataFrame(C.yield_dataset(1000))
