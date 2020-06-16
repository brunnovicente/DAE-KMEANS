import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from DAEKmeans import DAEKmeans
from sklearn.metrics import silhouette_score

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
sca = MinMaxScaler()

dados = pd.read_csv('D:/Drive UFRN/bases/mnist.csv')
#dados = dados[dados['classe'] < 5]
X = sca.fit_transform(dados.drop(['classe'], axis = 1).values)
Y = dados['classe'].values

daekmeans = DAEKmeans(784, 10, epocas=30)
daekmeans.fit(X)
preditas = daekmeans.predict(X)

s = silhouette_score(X, preditas)