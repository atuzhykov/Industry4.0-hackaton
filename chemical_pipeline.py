import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



input_file = "chemical_sensors_data.csv"
df = pd.read_csv(input_file, delimiter = ";", usecols=[3,4,5,6])

valid_data = []

for record in df.values:
    if 'None' not in record:
        valid_data.append(record)

valid_data = [list(map(float, sublist)) for sublist in valid_data]

train_data = valid_data[1:2000]
test_data = valid_data[2000:]



model = TSNE(learning_rate=100)
transformed = model.fit_transform(train_data)
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

plt.scatter(x_axis, y_axis)
plt.show()



