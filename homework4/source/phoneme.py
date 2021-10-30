"""
Author:
Date:
Description: Perceptron vs Logistic Regression on a Phoneme Dataset
"""

# utilities
from util import *

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import math
import pandas

######################################################################
# functions
######################################################################

def cv_performance(clf, X, y, kfs):
    """
    Determine classifier performance across multiple trials using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kfs        -- array of size n_trials
                      each element is one fold from model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_trials, n_fold)
                      each element is the (accuracy) score of one fold in one trial
    """

    #n_trials = len(kfs)
    #n_folds = kfs[0].n_splits
    #print(n_folds)
    scores = np.zeros((kfs, 2))
    #print(scores)

    ### ========== TODO: START ========== ###
    # part 2: run multiple trials of cross-validation (CV)
    # for each trial, get perf on 1 trial & update scores
    
    l = 0
    error1 = []
    for i in range(kfs):
        error = cv_performance_one_trial(clf, X, i,y)
        scores[i][0], scores[i][1] = i+1, error
        error1.append(error)
        
    
    
    r1 = np.mean(error1)
    r2 = np.std(error1)
    #print("Mean and Standard deviation for the classifier", r1 , r2)
    #print(" The test error for each trails")
    #print(scores)
    ### ========== TODO: END ========== ###

    return  r1, r2


def cv_performance_one_trial(clf, X, kf1,y):
    """
    Compute classifier performance across multiple folds using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kf         -- model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_fold, )
                      each element is the (accuracy) score of one fold
    """

    
    ### ========== TODO: START ========== ###
    # part 2: run one trial of cross-validation (CV)
    # for each fold, train on its data, predict, and update score
    # hint: check KFold.split and metrics.accuracy_score
    test_error = 0
    train_error = 0
    count = 0
    #x_tr,x_ts,y_tr,y_ts= train_test_split(traindata, y, test_size=0.3)
    kf = KFold(n_splits=10, random_state= kf1, shuffle=True)   
    #scores = np.zeros(kf.n_splits)#Implemetion of Kfold Cross validations
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print(train_index)
        #print(test_index)
        #print(count)
        count = count + 1
        training = clf.fit(X_train,y_train)
        pred = training.predict(X_test)
        test_error += (metrics.accuracy_score(y_test, pred))
    #print(test_index)
    #pred = training.predict(X_test)
    #test_error += (1-metrics.accuracy_score(y_test, pred))
    ### ========== TODO: END ========== ###
    #print(count)
    return test_error/count


######################################################################
# main
######################################################################

def main():
    np.random.seed(1234)

    #========================================
    # load data
    train_data = load_data('phoneme_train.csv')
    X = train_data.X
    y = train_data.y
    #print(X)
    #print(y)

    ### ========== TODO: START ========== ###
    # part 1: is data linearly separable? Try Perceptron and LogisticRegression
    # hint: set fit_intercept = True to have offset (i.e., bias)
    # hint: you may also want to try LogisticRegression with C=1e10
    x_tr,x_ts,y_tr,y_ts= train_test_split(X, y, test_size=0.2)
    clf1 = Perceptron(fit_intercept=True)
    clf2 = LogisticRegression(C=1e10)
    clf1.fit(x_tr, y_tr)
    clf2.fit(x_tr, y_tr)
#     plt.plot(x_tr, y_tr)
#     plt.show()
    predictions1 = clf1.predict(x_ts)
    ac = accuracy_score(y_ts,predictions1)
    print("Perceptron accuracy score")
    print(f'{ac:9.3f}')
    predictions2 = clf2.predict(x_ts)
    ac = accuracy_score(y_ts,predictions2)
    print("Logistic Regression accuracy score")
    print(f'{ac:9.3f}')
    


    ### ========== TODO: END ========== ###

    ### ========== TODO: START ========== ###
    # parts 3-4: compare classifiers
    # make sure to use same folds across all runs (hint: model_selection.KFold)
    # hint: for standardization, use preprocessing.StandardScaler()
    #cv_performance_one_trial(clf1, X, 3,y)
    kfs = 10
#     r1, r2 = cv_performance(clf1, X, y, kfs)
#     r3, r4 = cv_performance(clf2,X, y, kfs)
#     r5, r6 = cv_performance(clf2, X, y, kfs)
    print("classifier  |   µ    |   σ ")
    print("without preprossesing")
    #print("classifier µ σ")
    P = Perceptron(fit_intercept=True)
    L = LogisticRegression(fit_intercept = True,C=1e10)
    R = LogisticRegression(fit_intercept = True,penalty='l2', C=1, max_iter = 1000, solver = 
                          'lbfgs')
    
    r1, r2 = cv_performance(P, X, y, kfs)
    print(f'    P    {r1:9.3f} {r2:9.3f} ')
    r3, r4 = cv_performance(L, X ,y, kfs)
    print(f'    L    {r3:9.3f} {r4:9.3f} ')
    r5, r6 = cv_performance(R, X , y, kfs)
    print(f'    R    {r5:9.3f} {r6:9.3f} ')
    
    
    
    print("With Data standardizatin")
    #dataframe = pandas.read_csv(r'\File name.csv')
    # separate array into input and output components
#     X = array[:,0:8]
#     Y = array[:,8]
    
    scaler = StandardScaler() 
    X = scaler.fit_transform(X)
    #scaled = scaler.fit_transform(train_data)
    
    
    P = Perceptron(fit_intercept=True)
    L = LogisticRegression(fit_intercept = True,C=1e10)
    R = LogisticRegression(fit_intercept = True,penalty='l2',C=1)
    
    r1, r2 = cv_performance(P, X, y, kfs)
    print(f'    P    {r1:9.3f} {r2:9.3f} ')
    r3, r4 = cv_performance(L, X ,y, kfs)
    print(f'    L    {r3:9.3f} {r4:9.3f} ')
    r5, r6 = cv_performance(R, X , y, kfs)
    print(f'    R    {r5:9.3f} {r6:9.3f} ')
    
    
        
    ### ========== TODO: END ========== ###


if __name__ == '__main__':
    main()
