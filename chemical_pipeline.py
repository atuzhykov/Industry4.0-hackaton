import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans  
from data_fixer import data_fixer



input_file = "chemical_sensors_data.csv"
df = pd.read_csv(input_file, delimiter = ";", usecols=[3,4,5,6])

valid_data = data_fixer(df)

# model = TSNE(learning_rate=100)
# transformed = model.fit_transform(valid_data)
# x_axis = transformed[:, 0]
# y_axis = transformed[:, 1]

# plt.scatter(x_axis, y_axis)
# plt.show()


X = np.array(valid_data)
kmeans = KMeans(n_clusters=4)  
kmeans.fit(X) 
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow') 
plt.show() 





