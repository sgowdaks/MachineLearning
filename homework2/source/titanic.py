"""
Author:
Date:
Description:
"""

# Use only the provided packages!
import math
import csv
from util import *
from dtree import *
import numpy as np
#from numpy import apply
from sklearn.model_selection import KFold


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
        # insert your RandomClassifier code
        def probabal(x): #probability function
            return (x/n)
        
        vals, counts = np.unique(y, return_counts = True)
        n,d = X.shape
        result = list(map(probabal,counts))
        #print(result)
        res = {vals[i]: result[i] for i in range(len(counts))}   # setting the self.probabilities_ as a dictionery
        self.probabilities_ = res

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
        l = [0,1]
        n,d = X.shape
        y = np.random.choice(l, n, p = [self.probabilities_[0],self.probabilities_[1]]) 


        ### ========== TODO: END ========== ###

        return y


######################################################################
# functions
######################################################################

def error(clf, X, y, ntrials=100, test_size=0.2):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
        test_size   -- float (between 0.0 and 1.0) or int,
                       if float, the proportion of the dataset to include in the test split
                       if int, the absolute number of test samples

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO: START ========== ###
    # part b: compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    test_error = 0
    train_error = 0
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = ntrials)
        training = clf.fit(X_train,y_train)
        pred = training.predict(X_train)
        train_error += (1-metrics.accuracy_score(y_train, pred))
        pred = training.predict(X_test)
        test_error += (1-metrics.accuracy_score(y_test, pred))
    

    ### ========== TODO: END ========== ###

    return train_error/ntrials, test_error/ntrials



def write_predictions(y_pred, filename, yname=None):
    """Write out predictions to csv file."""
    out = open(filename, 'w')
    f = csv.writer(out)
    if yname:
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data('titanic_train.csv', header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    # training Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print('\t-- training error: %.3f' % train_error)
   



    ### ========== TODO: START ========== ###
    # part a: evaluate training error of Decision Tree classifier
    print('Classifying using Decision Tree...')
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1- metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    

    ### ========== TODO: END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import pydot
    from io import String
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames,
                         class_names=['Died', 'Survived'])
    graph = pydot.graph_from_dot_data(str(dot_data.getvalue()))[0]
    graph.write_pdf('dtree.pdf')
    """


    ### ========== TODO: START ========== ###
    # part b: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    clf = MajorityVoteClassifier() 
    train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.2)
    print(f"Manjority Vote Classifier train error = {train_error:.3f} and test error = {test_error:.3f}")
    clf =  RandomClassifier()
    train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.2)
    print(f"Ramdom Classifier train error = {train_error:.3f} and test error = {test_error:.3f}")
    clf =  tree.DecisionTreeClassifier(criterion='entropy')
    train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.2)
    print(f"Decision treeclassifier train error = {train_error:.3f} and test error = {test_error:.3f}")
    ### ========== TODO: END ========== ###



    ### ========== TODO: START ========== ###
    # part c: investigate decision tree classifier with various depths
    print('Investigating depths...')
    traine = []
    teste = []
    for i in range(1,21):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
        train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.2)
        traine.append(train_error)
        teste.append(test_error)
        
    x = np.arange(20) + 1   
    plt.plot(x, traine, label='Decision treeclassifier Training Error') 
    plt.plot(x, teste, label='Decision treeclassifier Testing Error')
    plt.xlabel('Maximum Depth') 
    plt.ylabel('Total Error') 
    plt.legend() 
    plt.plot()
    plt.grid()
    teste.sort()
    #print(teste[0])
    plt.plot(1, teste[0], 'bo')
    
   
    
    
    
    teste = []
    for i in range(1,21):
        clf = MajorityVoteClassifier()
        train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.2)
        teste.append(test_error)
        
    x = np.arange(20) + 1   
    plt.plot(x, teste, label='Majority Vote Classifire Testing Error')
    plt.xlabel('Maximum Depth') 
    plt.ylabel('Total Error') 
    plt.legend() 
    plt.plot()
    
    teste = []
    for i in range(1,21):
        clf = RandomClassifier()
        train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.2)
        teste.append(test_error)
        
    x = np.arange(20) + 1   
    plt.plot(x, teste, label='Random Classifier Testing Error')
    plt.title("Testing and Training error against Depths")
    plt.xlabel('Maximum Depth') 
    plt.ylabel('Total Error') 
    plt.legend() 
    plt.plot()
    fig = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    
        

    ### ========== TODO: END ========== ###



    ### ========== TODO: START ========== ###
    # part d: investigate decision tree classifier with various training set sizes
  
    
    print('Investigating training set sizes...')
    traine = []
    teste = []
    
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
    for i in np.arange(0.95,0.00,-0.05):
        train_error, test_error = error(clf, X, y, ntrials=100, test_size=i)
        traine.append(train_error)
        teste.append(test_error)
    x = np.arange(0.05,1.00,0.05)
    plt.plot(x, traine, label='Decision tree classifier Training Error') 
    plt.plot(x, teste, label='Decision tree classifier Testing Error')
    plt.xlabel('Training Data') 
    plt.ylabel('Error') 
    plt.legend() 
    plt.plot()
    plt.grid()
    
    traine = []
    teste = []
    clf = MajorityVoteClassifier()
    for i in np.arange(0.95,0.00,-0.05):
        train_error, test_error = error(clf, X, y, ntrials=100, test_size=i)
        traine.append(train_error)
        teste.append(test_error)
    x = np.arange(0.05,1.00,0.05)
    plt.plot(x, teste, label='Majority Vote classifier Testing Error')
    plt.legend() 
        
    traine = []
    teste = []
    clf = RandomClassifier()
    for i in np.arange(0.95,0.00,-0.05):
        train_error, test_error = error(clf, X, y, ntrials=100, test_size=i)
        traine.append(train_error)
        teste.append(test_error)
    x = np.arange(0.05,1.00,0.05)
    plt.plot(x, teste, label='Random classifier Testing Error')
    plt.legend()    
    plt.title("Testing and Training error along with the variation in Traning data")
        
        
        
        
        
        
    ax2 = plt.subplot(1, 1, 1)
    ### ========== TODO: END ========== ###



    ### ========== TODO: START ========== ###
    # Contest
    # uncomment write_predictions and change the filename
    #X = np.array(X)
    #print(X.shape)
    
    test_error = 0
    train_error = 0
    count = 0
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
    kf = KFold(n_splits=3, random_state=None, shuffle=True)                    #Implemetion of Kfold Cross validations
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        count = count + 1
        training = clf.fit(X_train,y_train)
        pred = training.predict(X_train)
        train_error += (1-metrics.accuracy_score(y_train, pred))
        pred = training.predict(X_test)
        test_error += (1-metrics.accuracy_score(y_test, pred))
        

    #print(count)
    train_error = test_error/count
    test_error = train_error/count
    print(f"Train error {train_error:.3f} and Test error {test_error:.3f} with the implementation of 3 Folds")



    # evaluate on test data
    titanic_test = load_data('titanic_test.csv', header=1, predict_col=None)
    X_test = titanic_test.X
    y_pred = clf.predict(X_test)
    #print(y_pred)
    #y_pred = training.predict(X_test)
    #print(y_pred)
    # take the trained classifier and run it on the test data
    #print(y_pred.shape)
    
    write_predictions(y_pred, 'data/shivani_titanic.csv', titanic.yname)

    ### ========== TODO: END ========== ###



    print('Done')


if __name__ == '__main__':
    main()
