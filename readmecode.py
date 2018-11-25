
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cgkit import CausalGraph

import statsmodels.api as sm
import statsmodels.formula.api as smf


C = CausalGraph(5,
                density =0.8,
                seed = 5,
                parameter_space = lambda: np.random.randint(-10,11)
                )
C.draw_graph()
#plt.savefig("figs/examplegraph.png")


(X,y), df = C.yield_dataset(1000)


with sns.color_palette("BrBG_d"):
    sns.pairplot(df)
#    plt.savefig('figs/pairs.png')

C.parameters

C.pintervene(effect = 4, cause = 0, parameter = 5)
(Xp,yp), dfp = C.yield_dataset(1000)


df['Intervention'] = 'No'
dfp['Intervention'] = 'Yes'

plot = pd.concat([df, dfp])

sns.pairplot(plot, hue = 'Intervention', palette = 'BrBG')
#plt.savefig("figs/pairs2.png")
