from sklearn.datasets import fetch_openml

#fetching the MNIST dataset from OpenLM.org

mnist = fetch_openml('minst_784', as_frame = False)

X, y = minst.data, minst.target

print("Shape of the minst dataset: \n", X.shape)
print("\nInfo of the minst dataset: \n", X.info())
print("\nDescription of the minst dataset: \n", X.describe())
print("\nMinst dataset: \n", X.head())
