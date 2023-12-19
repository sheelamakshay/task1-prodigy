import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target_names[iris.target]

# Display the first few rows of the dataset
print(data.head())

# Create the Bar Chart or Histogram
plt.figure(figsize=(10, 6))
plt.bar(data['target'], data['sepal length (cm)'])  # Choose a variable for the y-axis
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.title('Distribution of Sepal Length Across Iris Species')
plt.show()
