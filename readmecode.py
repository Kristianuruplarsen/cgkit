import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cgkit import CausalGraph

import statsmodels.api as sm
import statsmodels.formula.api as smf


C = CausalGraph(4,
                n_dummies = 1,
                density =0.8,
                seed = 5,
                parameter_space = lambda: np.random.randint(-10,11)
                )

C.draw_graph()
plt.savefig("figs/examplegraph.png")


(X,y), df = C.yield_dataset(1000)


with sns.color_palette("BrBG_d"):
    sns.pairplot(df)
    plt.savefig('figs/pairs.png')

C.pintervene(effect = 'c1', cause = 'c0', parameter = 5)
(Xp,yp), dfp = C.yield_dataset(1000)


df['Intervention'] = 'No'
dfp['Intervention'] = 'Yes'

plot = pd.concat([df, dfp])


sns.pairplot(plot, hue = 'Intervention', palette = 'BrBG')
plt.savefig("figs/pairs2.png")


from collections import defaultdict

result = sm.OLS(df['Y'], df.drop(['Y', 'Intervention'], axis = 1)).fit()

cint = result.conf_int()
est = result.params

act = C.parameters['Y']
act = defaultdict(int, act)

i = 0
for var in cint.index:
    if i == 0:
        plt.scatter([act[var]], [var], color = 'r', label = 'Truth')
        plt.scatter([est[var]], [var], facecolor = 'none', edgecolor = 'black', label = 'Mean estimate')
        plt.plot([cint[0][var], cint[1][var]], [var, var], color = "black", label = 'Confidence interval')
        i = 1
    else:
        plt.scatter([act[var]], [var], color = 'r')
        plt.scatter([est[var]], [var], facecolor = 'none', edgecolor = 'black')
        plt.plot([cint[0][var], cint[1][var]], [var, var], color = "black")

plt.legend()
plt.xlabel("Estimated value")
plt.ylabel("Variable")
plt.savefig("figs/regressionresults.png")
plt.show()


plt.scatter(
df[df['d0'] == 1]['Y'],
df[df['d0'] == 1]['c1'],
color = 'b'
)

plt.scatter(
df[df['d0'] == 0]['Y'],
df[df['d0'] == 0]['c1'],
color = 'r'
)
