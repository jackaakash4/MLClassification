import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml


#fetching the MNIST dataset from OpenLM.org

mnist = fetch_openml('mnist_784', as_frame = False)

X, y = mnist.data, mnist.target

print("Shape of the minst dataset: \n", X.shape)

#print("\nX: \n", X)

#Visualizing using maplotlib's imshow() function and using cmap = "binary" to get a grayscale color map where 0 is white and 255 is black

def plot_digit(image_data):
    image = image_data.reshape(28, 28)  #shapping it in  28 * 28 array
    plt.imshow(image, cmap = 'binary')
    plt.axis('off')
    
some_digit = X[0]
plot_digit(some_digit)
plt.show()


