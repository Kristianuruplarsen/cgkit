
import numpy as np
import networkx as nx
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from graph.causalgraph import CausalGraph


C = CausalGraph(5,
                density =0.8,
                seed = 1,
                parameter_space = lambda: np.random.randint(-10,11)
                )
C.draw_graph()
plt.savefig("figs/examplegraph.png")

raw = C.yield_dataset(1000)

df = pd.DataFrame(raw)
with sns.color_palette("GnBu_d"):
    sns.pairplot(df)
    plt.savefig('figs/pairs.png')


C.parameters

C.pintervene(effect = 0, cause = 2, parameter = -5)

df2 = pd.DataFrame(C.yield_dataset(nobs = 1000))

df['Intervention'] = 'No'
df2['Intervention'] = 'Yes'

df3 = pd.concat([df, df2])

sns.pairplot(df3, hue = 'Intervention', palette = 'GnBu_d')
plt.savefig("figs/pairs2.png")
