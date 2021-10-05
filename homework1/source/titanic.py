"""
Author: Shivani Gowda
Date: september 9th
Description: completed
"""

# Use only the provided packages!
import math
import csv
from util import *

from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object):
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier):

    def __init__(self):
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y):
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        vals, counts = np.unique(y, return_counts=True)
        #print(zip(vals,count))
        majority_val, majority_count = max(zip(vals, counts), key=lambda val_count: val_count[1])
        self.prediction_ = majority_val
        return self

    def predict(self, X):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None:
            raise Exception('Classifier not initialized. Perform a fit first.')

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier):

    def __init__(self):
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y):
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO: START ========== ###
        # part c: set self.probabilities_ according to the training set
        def probabal(x): #probability function
            return (x/n)
        
        vals, counts = np.unique(y, return_counts = True)
        n,d = X.shape
        result = list(map(probabal,counts))
        #print(result)
        res = {vals[i]: result[i] for i in range(len(counts))}   # setting the self.probabilities_ as a dictionery
        self.probabilities_ = res
        #print(list(results))

        ### ========== TODO: END ========== ###

        return self

    def predict(self, X, seed=1234):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None:
            raise Exception('Classifier not initialized. Perform a fit first.')
        np.random.seed(seed)

        ### ========== TODO: START ========== ###
        # part c: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        #n = np.random.seed(1234)
        l = [0,1]
        n,d = X.shape
        y = np.random.choice(l, n, p = [self.probabilities_[0],self.probabilities_[1]])  #using random.choice to set the y according the probablity distribution

        

        ### ========== TODO: END ========== ###

        return y


######################################################################
# functions
######################################################################

def plot_histogram(X, y, Xname, yname):
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    #print(targets) [0.0, 1.0]
    data = []; labels = []
    for target in targets:
        features = [X[i] for i in range(len(y)) if y[i] == target]
        #print(features) ---> assigning vales for each 1,0
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    #print(data)
    #print(labels)

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    #print(test_range) for each range bwt max and min
    #print(sorted(features))
    #print(test_range)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin #assiging the proper range if not
        #print(bins)
        #print("yes")
        align = 'left'
    else:
        bins = 10
        #print("no")
        align = 'mid'

    # plot
    plt.figure()
    n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
    plt.xlabel(Xname)
    plt.ylabel('Frequency')
    plt.legend() #plt.legend(loc='upper left')
    plt.show()


def plot_scatter(X, y, Xnames, yname):
    """
    Plots scatter plot of values in X grouped by y.

    Parameters
    --------------------
        X      -- numpy array of shape (n,2), feature values
        y      -- numpy array of shape (n,), target classes
        Xnames -- tuple of strings, names of features
        yname  -- string, name of target
    """

    # plot
    targets = sorted(set(y))
    #plt.figure()
    '''data = []
    labels = []
    for i in range(2,6,3):
        for target in targets:'''
        ### ========== TODO: START ========== ###
        # part b: plot
        # hint: use plt.plot (and set the label)
            
    
    color = ['r' if i == 0 else 'g' for i in y ]   # Green color represts 0 and red represents deaths
    labels = []
    labels.append('%s = %s' % ("red", "deaths"))
    labels.append('%s = %s' % ("Green", "survived" ))
    
    plt.scatter(X[:,5], X[:,2],linewidths=1, alpha=0.5,edgecolor='k', s = 50, color =  color)  #scatter plot function
    #plt.scatter(X[5],X[2])
        
        
        
        ### ========== TODO: END ========== ###
    plt.autoscale(enable=True)
    plt.xlabel(Xnames[5])
    plt.ylabel(Xnames[2])
    plt.legend()
    plt.show()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data('titanic_train.csv', header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    #print(X)
    #print(Xnames)
    y = titanic.y; yname = titanic.yname
    #print(y)
    #print(yname)
    n,d = X.shape  # n = number of examples, d =  number of features

     

    #========================================
    # plot histograms of each feature
    print('Plotting...')
    for i in range(d):
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
        #plot_hitogram




    ### ========== TODO: START ========== ###
    # part b: make scatterplot of age versus fare
    #for i in range(2,6,3):
    plot_scatter(X, y, Xnames, yname)  #calling the scatter plot function
    
     
    ### ========== TODO: END ========== ###



    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    #print(y_pred)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO: START ========== ###
    # part c: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO: END ========== ###



    print('Done')


if __name__ == '__main__':
    main()

