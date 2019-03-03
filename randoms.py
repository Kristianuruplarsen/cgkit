
# Preliminary work on a panel structure with potential
# drift in parameters over time.

#%%
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from cgkit import CausalGraph

#%%

C = CausalGraph(
                n_continuous = 5,
                n_dummies = 3,
                density =0.6,
                seed = 8,
                parameter_space = lambda: np.random.randint(-10,11),
                p_linear = 0.5
                )


C.draw_graph()

#%%
(X,y), df = C.yield_dataset(1000)

sns.pairplot(df)
plt.show()

#%%
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X = df.drop('Y', axis = 1).values
y = df['Y'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = MLPRegressor(hidden_layer_sizes=(200,100,50,100,))
#model = LinearRegression()
fm = model.fit(X,y)

p = fm.predict(X)

plt.scatter(p, y)
plt.plot(p,p, color = 'red')
#%%
