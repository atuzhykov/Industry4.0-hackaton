import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans  
from data_fixer import data_fixer



input_file = "chemical_sensors_data.csv"
df = pd.read_csv(input_file, delimiter = ";", usecols=[3,4,5,6])

valid_data = data_fixer(df)


fig = plt.figure()
ax = plt.axes(projection='3d')


X = np.array(valid_data)
kmeans = KMeans(n_clusters=4)  
kmeans.fit(X) 
ax.scatter(X[:,0],X[:,1],X[:,2], c=kmeans.labels_, cmap='rainbow') 
plt.show() 





