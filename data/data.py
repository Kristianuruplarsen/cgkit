
import numpy as np
import networkx as nx
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data.causalgraph import cgraph


C = cgraph(5, density =0.5)
C.draw_graph()
plt.savefig("figs/examplegraph.png")

raw = C.yield_dataset(1000)

df = pd.DataFrame(raw)
sns.pairplot(df)
plt.savefig('pairs.png')
