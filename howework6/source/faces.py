"""
Author: 
Date: 
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt

import util
# TODO: change cluster_soln to cluster if you do the extra credit
from cluster_5350.cluster import *


######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y):
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in range(n):
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images:
        count = 0
        for im in images[face]:
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def generate_points_2d(N, seed=1234):
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma):
        label += 1
        for i in range(N):
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# main
######################################################################

def main():
    ### ========== TODO: START ========== ###
    # part 1: explore LFW data set
    X, y = util.get_lfw_data()
#     print(X)
    #print(y)
    n, d = X.shape
    #print(n,d)
    #print(len(y))
    print("Random 6 images........")
    for i in range(6):
        raw = X[i]
        util.show_image(raw)
     
    main_average = []
    print("Averge of all images........")
    for i in range(d):
        raw = X[:,i]
        average = np.mean(raw,axis = 0)
        main_average.append(average)
        
    util.show_image(np.array(main_average) )
    #print(len(main_average))
    # TODO: display samples images and "average" image
    # TODO: display top 12 eigenfaces
    U, mu = util.PCA(X)
    n,d = U.shape
    #print(n,d)
    print("Top 12 eigen vectors................")
    util.plot_gallery([util.vec_to_image(U[:,i]) for i in range(12)])
    print("For 2 eigen faces...............")
    for i in range(2):
        raw = U[:,i]
        util.show_image(util.vec_to_image(raw))
    # TODO: try lower-dimensional representations
    print("Applying PCA and reconstructing it back")
    List = [1, 10, 50, 100, 500, 1288]
    for l in List:
        Z, Ul = util.apply_PCA_from_Eig(X, U, l, mu)
        X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
        print("For L of ",l)
        util.plot_gallery(X_rec[:12])
    

    ## ========== TODO: END ========== ###
    
    
#     ===============================================
#     (Optional) part 2: test Cluster implementation    
#     centroid: [ 1.04022358  0.62914619]
#     medoid:   [ 1.05674064  0.71183522]
    
#     np.random.seed(1234)
#     sim_points = generate_points_2d(20)
#     cluster = Cluster(sim_points)
#     print('centroid:', cluster.centroid().attrs)
#     print('medoid:', cluster.medoid().attrs)
    
#     # part 2: test kMeans and kMedoids implementation using toy dataset
#     np.random.seed(1234)
#     sim_points = generate_points_2d(20)
#     k = 3
    
#     # cluster using random initialization
#     kmeans_clusters = kMeans(sim_points, k, init='random', plot=True)
#     kmedoids_clusters = kMedoids(sim_points, k, init='random', plot=True)
    
#     # cluster using cheat initialization
#     kmeans_clusters = kMeans(sim_points, k, init='cheat', plot=True)
#     kmedoids_clusters = kMedoids(sim_points, k, init='cheat', plot=True)    
    
    
    ## ========== TODO: START ========== ###    
    #part 3a: cluster faces
    np.random.seed(1234)
    k = 4
    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 15], 40)
    points = build_face_image_points(X1, y1)
    #print(X1.shape)
    #print(points)
    k_means = []
    k_medoids = []
    
    for i in range(10):
        kmeans = kMeans(points, k, init="random")
        k_means.append(kmeans.score())

    Mean = np.mean(k_means)
    Min = min(k_means)
    Max = max(k_means)
    
    print("Mean, Min and Max for Kmeans are : ",Mean,Min,Max)
        
    for i in range(10):
        kmedoids = kMedoids(points, k, init="random")
        k_medoids.append(kmedoids.score())
        
    Mean = np.mean(k_medoids)
    Min = min(k_medoids)
    Max = max(k_medoids)
    
    print("Mean, Min and Max for kmedoids are : ", Mean, Min, Max)
    #print(Mean,Min,Max)
        
        
    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)
    k = 2
    U, mu = util.PCA(X)
    X1, y1 = util.limit_pics(X, y, [3, 12], 40)
    #points = build_face_image_points(X1, y1)
    
    l = np.arange(1,42,2)
    #U, mu = util.PCA(X1)
    
    k_means = []
    k_medoids = []
    #print(init="cheat")
    for i in l:
        Z, Ul = util.apply_PCA_from_Eig(X1, U, i, mu)
        points = build_face_image_points(Z, y1)
        kmeans = kMeans(points, k, init='cheat')
        k_means.append(kmeans.score())
        kmedoids = kMedoids(points,k, init='cheat')
        k_medoids.append(kmedoids.score())
        
        
#     plt.plot(l,k_means)
#     print(k_means)
#     plt.plot(l,k_medoids)
#     print(k_medoids)
    scatter1 = plt.scatter(l, k_means)
    scatter2 = plt.scatter(l, k_medoids)
    plt.suptitle('k_means and k_medoids')
    plt.xlabel('L')
    plt.ylabel('Score')
    plt.legend((scatter1, scatter2),
               ('k_means', 'k_medoids'))
    plt.show()
        
    
    
#     print(l)
    
    # part 3c: determine "most discriminative" and "least discriminative" pairs of images
    np.random.seed(1234)
    n = 16
    best = 0
    worst = 100000
    bestPair = None
    worstPair = None
    for p1 in range(n):
        for p2 in range(p1+1, n):
            X3, y3 = util.limit_pics(X, y, [p1, p2], 40)
            points = build_face_image_points(X3, y3)
            Kmeans = kMeans(points, 2, init="cheat")
            score = Kmeans.score()
            if score > best:
                best = score
                bestpair = (p1, p2)
            if score < worst:
                worst = score
                worstpair = (p1, p2)
                
    print("Best pair with the score : ",bestpair,best)
    s = list(bestpair)
    X3, y3 = util.limit_pics(X, y, s , 40)
    for i in range(2):
        raw = X3[i]
        util.show_image(raw)

    print("Worst pair with the score : ",worstpair,worst)
    s = list(worstpair)
    X3, y3 = util.limit_pics(X, y, s , 40)
    for i in range(2):
        raw = X3[i]
        util.show_image(raw)

    #util.plot_gallery([util.vec_to_image() for i in range(12)])
    
   
    
            

    
    
    ### ========== TODO: END ========== ###


if __name__ == "__main__":
    main()