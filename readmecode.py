
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
plt.savefig("figs/examplegraph.png")


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





result = sm.OLS(df['Y'], df[[0,1,2,3,4]]).fit()

cint = result.conf_int()
est = result.params
act = {0:0, 1:-3, 2:0, 3:8, 4:6}

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
plt.yticks([0,1,2,3,4])
plt.xlabel("Estimated value")
plt.ylabel("Variable")
plt.savefig("figs/regressionresults.png")
plt.show()
