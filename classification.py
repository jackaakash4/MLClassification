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
print("Digit is: \t", y[0])

#splitting the training and testing sets 
#since mnist already splited it 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#Training a binary classifiers
#now only trying to identify only one digit i.e 5. so it is capable of distinguishing between just two classes 5 or not 5

y_train_5 = (y_train == '5')        #True for all 5s, and False for all other digits
y_test_5 = (y_test == '5')

#starting SGD i.e stochastic gradient descent
#creating SGD

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)

#now it is used to detect images of the number 5

is5 = sgd_clf.predict([some_digit])       #some digit is 5 here

print("Is it 5 ?", is5)

#since it predited right
#now measure the performance measure
#using cross_val_score() to measure this classifier
#it uses k-fold cross-validation with three folds
#splitting the training set to k folds(in this case three)
#then training the model k times, holding out a different fold each time for evaluation

from sklearn.model_selection import cross_val_score
print("Cross value score of SGD classifier is : \t", cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = 'accuracy'))












