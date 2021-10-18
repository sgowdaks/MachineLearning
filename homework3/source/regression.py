"""
Author     : 
Date       : 
Description: Polynomial Regression
This code was adapted from course material by Jenna Wiens (UMichigan).
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
import timeit
import pandas as pd


######################################################################
# classes
######################################################################

class Data:
    
    def __init__(self, X=None, y=None):
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    

    def load(self, filename):
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)
        
        # load data
        with open(f, 'r') as fid:
            data = np.loadtxt(fid, delimiter=",")
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]
    

    def plot(self, **kwargs):
        """Plot data."""
        
        if 'color' not in kwargs:
            kwargs['color'] = 'b'
        
        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()


# wrapper functions around Data class
def load_data(filename):
    data = Data()
    data.load(filename)
    return data


def plot_data(X, y, **kwargs):
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression():
    
    def __init__(self, m=1, reg_param=0,ite= 0):
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param
        self.iter = ite
    
    
    def generate_polynomial_features(self, X):
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """
        
        n,d = X.shape
        #print(X.shape)
        
        ### ========== TODO: START ========== ###
        # part b: modify to create matrix for simple linear model
        # hint: use np.ones(), np.append(), and np.power()
 
        m = self.m_
        #print(m)
        if m == 1:
            Phi = np.zeros((n,2))
            for i in range(n):
                Phi[i,0] = 1
                Phi[i, 1] = X[i]
                
        else:
            Phi = np.ones((n, 1))
            for i in range(1, m+1):
                Phi = np.append(Phi, np.power(X, i), 1)
            
            
            
                
        
            
        # part g: modify to create matrix for polynomial model
        #Phi = X
        #print(Phi)
        
        ### ========== TODO: END ========== ###
        
        return Phi
        #return X
    
    
    def fit_SGD(self, X, y, eta,
                eps=1e-10, tmax=1_000_000, verbose=False):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares stochastic gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size (also known as alpha)
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0:
            raise Exception('SGD with regularization not implemented')
        
        if verbose:
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()
        X_orig = X
        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)  # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration
        t = 0
        
        # SGD loop
        with tqdm(range(tmax), disable=True) as pbar:
            for t in pbar:
                ### ========== TODO: START ========== ###
                # part f: update step size
                # change the default eta in the function signature to 'eta=None'
                # and update the line below to your learning rate function
                if eta_input is None:
                    eta = 1/(1+float(t))
                    #print("this eta was excecuted")
                    #eta = None # change this line
                else:
                    eta = eta_input
                ### ========== TODO: END ========== ###

                # iterate through examples
                for i in range(n):
                    ### ========== TODO: START ========== ###
                    phi = X[i,:]
                    #print(phi)
                    #print("*")
                    y_pred = np.dot(self.coef_, phi)
                    print(y_pred)
                    print("*")
                    # part d: update theta (self.coef_) using one step of SGD
                    k = (y_pred - y[i,])*phi
                    #print(k)
                    self.coef_ = self.coef_ - eta*(y_pred - y[i,])*phi
                    #print(self.coef_)
                #print("**")

                # hint: you can simultaneously update all theta via vector math
                # track error
                # TODO: predict y using updated theta
                y_pred = self.predict(X_orig)
                err = err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n) 
                pbar.set_postfix_str(f'Error:{err:g}')
                ### ========== TODO: END ========== ###

                # stop?
                #print(err_list)
                if t > 0 and abs(err_list[t] - err_list[t-1]) < eps:
                    break

                # debugging
                if verbose and t % 100 == 0:
                    x = np.reshape(X[:,1], (n,1))
                    cost = self.cost(x,y)
                    plt.subplot(1, 2, 1)
                    plt.cla()
                    plot_data(x, y)
                    self.plot_regression()
                    plt.subplot(1, 2, 2)
                    plt.plot([t+1], [cost], 'bo')
                    plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                    plt.draw()
                    plt.pause(0.05) # pause for 0.05 sec
        
        
        #print(len(err_list))
        err = err_list.reshape(-1)
        err = err[:t+1]
        #print(len(err1))
        #print(len(err_list))
        xx = np.arange(len(err))
        if eta_input == None:
            plt.plot(xx,err, label='updated Leraning Rate') 
        else:
            plt.plot(xx,err, label='Leraning Rate %f' % eta_input)
        plt.xlabel("iterations")
        plt.ylabel("error")
        plt.title("error vs iterations")
        #plt.show()
        plt.legend()
        print('number of iterations: %d' % (t+1))
        self.ite = t+1
        
        return self
    
    
    def fit(self, X, y):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
                
        Returns
        --------------------        
            self    -- an instance of self
        """
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO: START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        # be sure to update self.coef_ with your solution
#         with tqdm(range(1)) as pbar:
#             for i in pbar:
#                 X_transp_x= np.matmul(X.T, X)
#                 X_transp_x_inv= np.linalg.inv(X_transp_x)
#                 self.coef_= np.matmul(np.matmul(X_transp_x_inv, X.T), y)
        
        X_transp_x= np.matmul(X.T, X)
        X_transp_x_inv= np.linalg.inv(X_transp_x)
        self.coef_= np.matmul(np.matmul(X_transp_x_inv, X.T), y)
        
        







        
        
                
        ### ========== TODO: END ========== ###
    
    
    def predict(self, X):
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None:
            raise Exception('Model not initialized. Perform a fit first.')
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO: START ========== ###
        # part c: predict y (hint: X times theta)
        coef = self.coef_
        y = np.dot(coef, np.transpose(X))
        
        #y = None
        ### ========== TODO: END ========== ###
        
        return y
    
    
    def cost(self, X, y):
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO: START ========== ###
        # part d: get predicted y, then compute J(theta)
        y_pred = self.predict(X)
        co = (1/2)*((y_pred - y)**2)
        cost = np.sum(co)
        ### ========== TODO: END ========== ###
        return cost
    
    
    def rms_error(self, X, y):
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO: START ========== ###
        # part h: compute RMSE
        #error = np.mean((self.cost(X,y)-y) ** 2)
        #print(error)
        error = np.sqrt((2*self.cost(X, y))/np.shape(y))
        ### ========== TODO: END ========== ###
        return error
    
    
    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs):
        """Plot regression line."""
        if 'color' not in kwargs:
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = '-'
        
        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main():
    # toy data
    X = np.array([2]).reshape((1,1))          # shape (n,d) = (1L,1L)
    y = np.array([3]).reshape((1,))           # shape (n,) = (1L,)
    #print(X.shape,y.shape)
    coef = np.array([4,5]).reshape((2,))      # shape (d+1,) = (2L,), 1 extra for bias
    
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')
    #print(train_data.X)

    
    
    ### ========== TODO: START ========== ###
    #part a: plot train and test data
    print('Visualizing data...')
    #train_d = train_data.X.reshape((20,))
    #print(train_d)
    X = train_data.X
    y = train_data.y
    plt.scatter(train_data.X.reshape((20,)), train_data.y)
    plt.xlabel("train_X")
    plt.ylabel("train_y")
    plt.title("X_train and y_train")
    plt.show()
    
    plt.scatter(test_data.X.reshape(20,),test_data.y)
    plt.xlabel("test_X")
    plt.ylabel("test_y")
    plt.title("X_test and y_test")
    plt.show()
    
    ### ========== TODO: END ========== ###
    
    
    ### ========== TODO: START ========== ###
    # parts b-f: main code for linear regression
    print('Investigating linear regression...')
    
    # model
    model = PolynomialRegression()
    
    # test part b -- soln: [[1 2]]
    print(model.generate_polynomial_features(X))
    print("\n")
    
    # test part c -- soln: [14]
    model.coef_ = coef
    print("coef: ",model.predict(X))
    
    print("\n")
    
    # test part d, bullet 1 -- soln: 60.5
    print("cost function:",model.cost(X, y))
    
    print("\n")
    
    # test part d, bullets 2-3
    # for eta = 0.01, soln: theta = [2.44; -2.82], iterations = 616
    start = timeit.default_timer()
    model.fit_SGD(train_data.X, train_data.y, eta = 0.01, verbose= False)
    stop = timeit.default_timer()
    print('sgd solution: %s' % str(model.coef_))
    print('Time for sgd solution: ', stop - start)  
    
    print("\n")
    print("table for varying alpha size")
    list1 = [0.0001, 0.001, 0.01, 0.1]
    list2 = []
    list3 = []
    for i in list1:
        model.fit_SGD(train_data.X, train_data.y, i)
        list2.append(model.ite)
        list3.append(str(model.coef_))
    #print(list2)
    print("\n")
    d = {'alpha': list1, 'iterations': list2, 'coef': list3}
    df = pd.DataFrame(data=d)
    print(df)
    
    print("\n")
    
    
    
        
    
    # test part e -- soln: theta = [2.45; -2.82]
    start = timeit.default_timer()
    model.fit(train_data.X, train_data.y)
    stop = timeit.default_timer()
    print('closed_form solution: %s' % str(model.coef_))
    print('Time for closed_form solution: ', stop - start)
    
    print("\n")
    
    # non-test code (YOUR CODE HERE)
    
    ### ========== TODO: END ========== ###
    
    #part f
    print("Results with eta = None")
    model.fit_SGD(train_data.X, train_data.y,eta = None,verbose= False)
    print('sgd solution after predicting tmax: %s' % str(model.coef_))
    print("\n")   
    
    
    
    ### ========== TODO: START ========== ###
    # parts g-i: main code for polynomial regression
    print('Investigating polynomial regression...')
    
    # toy data
    m = 2                                     # polynomial degree
    coefm = np.array([4,5,6]).reshape((3,))   # shape (3L,), 1 bias + 3 coeffs
        
    # test part g -- soln: [[1 2 4]]
    model = PolynomialRegression(m)
    print("Polynomial feature:",model.generate_polynomial_features(X))
    print("\n")
    
    # test part h -- soln: 35.0
    model.coef_ = coefm
    print("RMS_error = ",model.rms_error(X, y))
    print("\n")
    
    # non-test code (YOUR CODE HERE)
        
    ### ========== TODO: END ========== ###
    
    
    ### ========== TODO: START ========== ###
    # parts j-k: main code for regularized regression
    print('Investigating regularized regression...')
    
    # test part j -- soln should be close to [3 0 0]
    # note: your solution may be slightly different
    #       due to limitations in floating point representation
    RMS_error_train = []
    RMS_error_test = []
    for i in range(0,11):
        model = PolynomialRegression(m=i)
        model.fit(X, y)
        RMS_error_train.append(model.rms_error(X, y))
        model.fit(test_data.X, test_data.y)
        RMS_error_test.append(model.rms_error(X, y))
        
    print("RMS train minimum error: %f",min(RMS_error_train))
    print("RMS test minimum error: %f",min(RMS_error_test))
    #print("10th order is giving the lowerst RMS error")
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,11),RMS_error_train,label="RMS train Error")
    ax.plot(np.arange(0,11),RMS_error_test,label="RMS test error")
    ax.legend()
    ax.set_title('RMS train and error vs n polynomial')
    plt.xlabel("Increasing polynomial order")
    plt.ylabel("RMS error")
    plt.show()
    
    print("3rd order is giving the lowerst RMS error")
    
    

    
        
    
    
    
    # non-test code (YOUR CODE HERE)
        
    ### ========== TODO: END ========== ###
    
    print('Done!')


if __name__ == "__main__":
    main()