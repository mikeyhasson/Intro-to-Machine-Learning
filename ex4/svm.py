#################################
# Your name:Michal Hasson
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    C=1000
    SV_number =[]
    for kernel_name in ['linear', 'quadratic', 'rbf']:
        if kernel_name == 'quadratic':
            kernel = svm.SVC(C=C,kernel='poly',degree=2,coef0=1)
        else:
            kernel = svm.SVC(C=C, kernel=kernel_name)
        kernel.fit(X_train,y_train)
        SV_number.append(kernel.n_support_)
        print(kernel_name+":",kernel.n_support_)
        create_plot(X_train, y_train, kernel)
        #plt.show()

    return np.array(SV_number)




def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """

    val_scores = []
    train_scores=[]
    vals =[10**x for x in range(-5, 6)]
    for val in vals:
        kernel = svm.SVC(C=val, kernel='linear')
        kernel.fit(X_train, y_train)
        train_scores.append(kernel.score(X_train, y_train))
        val_scores.append(kernel.score(X_val, y_val))



    print("Best C is:", vals[np.argmax(val_scores)])
    plt.plot(vals, val_scores,color='green',label='Validation accuracy')
    plt.plot(vals, train_scores,color='blue',label='Training accuracy')
    plt.xlim((vals[0],vals[-1]))
    plt.xlabel("C")
    plt.ylabel("Accuracy precentage")
    plt.xscale('log')
    plt.legend()

    return np.array(val_scores)



def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    val_scores = []
    train_scores=[]
    C=10
    vals =[10**x for x in range(-5, 6)]
    for val in vals:
        kernel = svm.SVC(C=C, kernel='rbf',gamma=val)
        kernel.fit(X_train, y_train)
        train_scores.append(kernel.score(X_train, y_train))
        val_scores.append(kernel.score(X_val, y_val))



    print("Best Gamma is:", vals[np.argmax(val_scores)])
    plt.plot(vals, val_scores,color='green',label='Validation accuracy')
    plt.plot(vals, train_scores,color='blue',label='Training accuracy')
    plt.xlim((vals[0],vals[-1]))
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy precentage")
    plt.xscale('log')
    plt.legend()

    return np.array(val_scores)
def graphs():
    for val in [1,100,100000]:
        kernel = svm.SVC(C=val, kernel='linear')
        kernel.fit(X_train, y_train)
        create_plot(X_val, y_val, kernel)
        plt.title("C=" + str(val))
        plt.show()

    # Q3

    for val in [0.001,0.1,1,1000]:
        kernel = svm.SVC(C=10, kernel='rbf', gamma=val)
        kernel.fit(X_train, y_train)
        create_plot(X_val, y_val, kernel)
        plt.title("Gamma="+ str(val))
        plt.show()

if __name__ == "__main__":
    X_train, y_train, X_val, y_val=get_points()
    graphs()
    train_three_kernels(X_train, y_train, X_val, y_val)
    linear_accuracy_per_C(X_train, y_train, X_val, y_val)
    plt.show()
    rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)
    plt.show()

