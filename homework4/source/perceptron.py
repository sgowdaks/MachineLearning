"""
Author:
Date:
Description: Perceptron
"""

# This code was adapted course material by Tommi Jaakola (MIT).

# utilities
from util import *

# scikit-learn libraries
from sklearn.svm import SVC

######################################################################
# functions
######################################################################

def load_simple_dataset(start=0, outlier=False):
    """Simple dataset of three points."""

    #  dataset
    #     i    x^{(i)}      y^{(i)}
    #     1    (-1, 1)^T    1
    #     2    (0, -1)^T    -1
    #     3    (1.5, 1)^T   1
    #   if outlier is set, x^{(3)} = (12, 1)^T

    # data set
    data = Data()
    data.X = np.array([[ -1, 1],
                       [  0,-1],
                       [1.5, 1]])
    if outlier:
        data.X[2,:] = [12, 1]
    data.y = np.array([1, -1, 1])

    # circularly shift the data points
    data.X = np.roll(data.X, -start, axis=0)
    data.y = np.roll(data.y, -start)

    return data


def plot_perceptron(data, clf, plot_data=True, axes_equal=False, **kwargs):
    """Plot decision boundary and data."""
    assert isinstance(clf, Perceptron)
    plt.subplot()
    # plot options
    if 'linewidths' not in kwargs:
        kwargs['linewidths'] = 2
    if 'colors' not in kwargs:
        kwargs['colors'] = 'k'

    # plot data
    if plot_data: data.plot()

    # axes limits and properties
    xmin, xmax = data.X[:, 0].min() - 1, data.X[:, 0].max() + 1
    ymin, ymax = data.X[:, 1].min() - 1, data.X[:, 1].max() + 1
    if axes_equal:
        xmin = ymin = min(xmin, ymin)
        xmax = ymax = max(xmax, ymax)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    # create a mesh to plot in
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

    # determine decision boundary
    #print('>> ravel ', xx.ravel(), yy.ravel())
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # plot decision boundary
    Z = Z.reshape(xx.shape)
    CS = plt.contour(xx, yy, Z, [0], **kwargs)

    # legend
    if 'label' in kwargs:
        #plt.clabel(CS, inline=1, fontsize=10)
        CS.collections[0].set_label(kwargs['label'])

    plt.show()


######################################################################
# classes
######################################################################

class Perceptron:

    def __init__(self):
        """
        Perceptron classifier that keeps track of mistakes made on each data point.

        Attributes
        --------------------
            coef_     -- numpy array of shape (d,), feature weights
            mistakes_ -- numpy array of shape (n,), mistakes per data point
        """
        self.coef_ = None
        self.mistakes_ = 0

    def fit(self, X, y, coef_init=None, verbose=False):
        """
        Fit the perceptron using the input data.

        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
            y         -- numpy array of shape (n,), targets
            coef_init -- numpy array of shape (n,d), initial feature weights
            verbose   -- boolean, for debugging purposes

        Returns
        --------------------
            self      -- an instance of self
        """
        # get dimensions of data
        n,d = X.shape
        # initialize weight vector to all zeros
        if coef_init is None:
            self.coef_ = np.zeros(d)
        else:
            self.coef_ = coef_init

        # record number of mistakes we make on each data point
        self.mistakes_ = 0
        # debugging
        if verbose:
            print(f'\ttheta^0 = {self.coef_}')

        ### ========== TODO: START ========== ###
        # part 1: implement perceptron algorithm
        # cycle until all examples are correctly classified
        # do NOT shuffle examples on each iteration
        # on a mistake, be sure to update self.mistakes_ and self.coef_
        #               and if verbose, output the updated self.coef_
        if(coef_init==None):
            for _ in range(5):
                  for i in range(n):
                    if y[i] * np.dot(X[i,:],self.coef_) <= 0:
                        self.coef_ = self.coef_ + y[i]*X[i,:]
                        self.mistakes_ = self.mistakes_ +1 

        else:
            n_epoch = 100
            for _ in range(n_epoch):
                  for i in range(n):
                    if y[i] * np.dot(X[i,:],self.coef_) <= 0:
                        self.coef_ = self.coef_ + y[i]*X[i,:]
                        self.mistakes_ = self.mistakes_ +1 
                    
                    
                
                    
                
                
      
            
            
            
            
            
            
        

        ### ========== TODO: END ========== ###

        return self

    def predict(self, X):
        """
        Predict labels using perceptron.

        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features

        Returns
        --------------------
            y_pred    -- numpy array of shape (n,), predictions
        """
#         print("*)=")
#         print(np.sign(np.dot(X, self.coef_)))
#         print('>>coef', self.coef_)
#         print('>>X', X)    
        return np.sign(np.dot(X, self.coef_))


def load_data(filename):
    data = Data()
    data.load(filename)
    return data


######################################################################
# main
######################################################################

def main():

    #========================================
    # test simple data set

    # starting with data point $x^{(1)}$ without outlier
    #   coef = [ 0.  1.], mistakes = 1
    # starting with data point $x^{(2)}$ without outlier
    #   coef = [ 0.5  2. ], mistakes = 2
    # starting with data point $x^{(1)}$ with outlier
    #   coef = [ 0.  1.], mistakes = 1
    # starting with data point $x^{(2)}$ with outlier
    #   coef = [ 6.  7.], mistakes = 7
    clf = Perceptron()
    
    for outlier in (False, True):
        for start in (1, 2):
            text = 'starting with data point $x^{(%d)}$ %s outlier' % \
                (start, 'with' if outlier else 'without')
            print(text)
#             k = plot_perceptron(data, clf, plot_data=True, axes_equal=False, **kwargs)
#             plt.figure() 
            
           
            
            data = load_simple_dataset(start, outlier)
            #print(data)
#             k = plot_perceptron(data, clf, plot_data=True, axes_equal=False)
#             plt.figure() 
            clf.fit(data.X, data.y, verbose=False)
            #plt.show()
            plt.title(text)
            print(f'\tcoef = {clf.coef_}, mistakes = {clf.mistakes_}')
            plot_perceptron(data, clf, plot_data=True, axes_equal=False)
    
    ### ========== TODO: START ========== ###
    # part 2: train a perceptron with two initializations (hint: use coef_init)
    # What are the final coefficients after convergence? Are they the same?
    train_data = load_data('perceptron_data.csv') 
    X = train_data.X
    y = train_data.y
#     print(X)
#     print(y)
    weights = [[0,0],[1,0]]
    #print(weights)
    print('\n')
    for i in range(2):
        weight = weights[i]
        print("For ",weight)
        clf.fit(data.X, data.y, weight, verbose=False)
        print("Inital Coef: ", weight)
        print(f'\tcoef = {clf.coef_}, mistakes = {clf.mistakes_}')
        plot_perceptron(train_data, clf, plot_data=True, axes_equal=False)
        
        
    ### ========== TODO: END ========== ###


if __name__ == '__main__':
    main()
