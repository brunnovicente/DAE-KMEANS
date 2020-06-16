from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from sklearn.cluster import KMeans
import numpy as np

class DAEKmeans:
    
    def __init__(self, entrada, k, epocas=200, lote=500):
        
        self.k = k
        self.entrada = entrada
        self.epocas = epocas
        self.lote = lote
        
        input_img = Input(shape=(self.entrada,))
        encoded = Dense(512, activation='relu')(input_img)
        drop = Dropout(0.2)(encoded)
        encoded = Dense(256, activation='relu')(drop)
        drop = Dropout(0.2)(encoded)
        encoded = Dense(128, activation='relu')(drop)
        
        Z = Dense(self.k, activation='relu')(encoded)
        
        decoded = Dense(10, activation='relu')(Z)
        drop = Dropout(0.2)(decoded)
        decoded = Dense(128, activation='relu')(drop)
        drop = Dropout(0.2)(decoded)
        decoded = Dense(255, activation='relu')(drop)
        drop = Dropout(0.2)(decoded)
        decoded = Dense(512, activation='relu')(drop)
        decoded = Dense(self.entrada, activation='sigmoid')(decoded)
                        
        self.encoder = Model(input_img, Z)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.summary()
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=0.1, decay=0, momentum=0.9))
        
        self.kmeans = KMeans(n_clusters = self.k)
    
    def fit(self, X):
        self.autoencoder.fit(X, X, epochs=self.epocas, batch_size=self.lote)
        Z = self.encoder.predict(X)
        self.kmeans.fit(Z)
        
    def predict(self, X):
        return self.kmeans.predict(self.encoder.predict(X))
        
        