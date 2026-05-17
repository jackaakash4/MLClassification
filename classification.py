from sklearn.datasets import fetch_openml

#fetching the MNIST dataset from OpenLM.org

mnist = fetch_openml('mnist_784', as_frame = False)

X, y = mnist.data, mnist.target

print("Shape of the minst dataset: \n", X.shape)

print("\nX: \n", X)
