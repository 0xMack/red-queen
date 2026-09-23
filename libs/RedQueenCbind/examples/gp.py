# print(redqueen.__file__)
# print(dir(redqueen))

from redqueen import Population
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Convert to NumPy arrays
X = iris.data          # features
y = iris.target        # labels
feature_names = iris.feature_names
target_names = iris.target_names

print(X.shape)  # (150, 4)
print(y.shape)  # (150,)
print(X.tolist())
# print(Population(X.tolist()))
pop = Population(X)
print(pop.predict().shape)