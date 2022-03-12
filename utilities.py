'''
UTILITIES AND PLOTTING
---------

These are all the functions thatget some job done without leading to a deeper
understanding of the overall method.

---------
'''
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import DBSCAN
import matplotlib as mpl
import configparser
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def unit_vector(vector):
    """ Normalizes a given vector.  """
    return vector / np.linalg.norm(np.array(vector))


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'. Example:
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
            
    Parameters:
    -----------
    Input:
    v1, v2 - array. Two vectors.
    
    Output:
    angle: float. Angle between the two vectors, in radians.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle


def yes_or_no(question):
    """ Poses a question to the user and waits for either a y/n """
    while "the answer is invalid":
        reply = str(input(question+' (y/n) ')).lower().strip()
        if reply[0] == "y":
            return True
        if reply[0] == "n":
            return False


def get_filename_parts(inputfile):
    """
    Cut the fname string into 3 parts: path+name+extension.
    
    Parameters:
    -----------
    Input:
    inputfile - string. Path + filename of the image.
    
    Output:
    path, fname, extension - strings. 
    """
    extension_start = [m.start() for m in re.finditer("\.",inputfile)][-1]
    try:
        fname_start = [m.start() for m in re.finditer("/",inputfile)][-1]
        path = inputfile[:fname_start+1]
    except IndexError:
        fname_start = -1
        path = "/"
    fname = inputfile[fname_start+1:extension_start]
    extension = inputfile[extension_start:]
    return path, fname, extension


def getXYSfromKPS(kps):
    """
    Get the attributes corresponding to x/y-coordinates and sizes of the keypoints kps.
    
    Parameters:
    ----------
    Input:
    kps - tuple. Keypoints, obtained through cv2.xfeatures2d.SIFT_create(...).detect(...).
    
    Output:
    x ,y, sizes - arrays. Coordinates and sizes of the keypoints.
    """
    n_kp = np.size(kps)
    x = np.zeros(n_kp)
    y = np.zeros(n_kp)
    sizes = np.zeros(n_kp)
    for i in range(n_kp):
        x[i], y[i] = kps[i].pt
        sizes[i] = kps[i].size
    x ,y, sizes = np.array(x), np.array(y), np.array(sizes)
    return x, y, sizes


def get_correctGreyCmap(name):
    """
    The colormap 'Greys_r' in matplotlib doesn't perfectly recreate the correct greyscale values
    which can lead to a loss of contrast. Instead a LinearSegmentedColormap should work better.
    """
    colors_dict = {"red":   [(0.0, 0.0, 0.0),
                             (1.0, 1.0, 1.0)],
                   "green": [(0.0, 0.0, 0.0),
                             (1.0, 1.0, 1.0)],
                   "blue":  [(0.0, 0.0, 0.0),
                             (1.0, 1.0, 1.0)]}
    return mpl.colors.LinearSegmentedColormap(name, colors_dict)        


def plot_clusters(img, kps, labels, colormap='jet', **kwargs):
    """
    Plots on top of the image all the keypoints colored according to their label.
    
    Parameters:
    -----------
    Input:
    img - array. Input image.
    kps - tuple. Keypoints, obtained through cv2.xfeatures2d.SIFT_create(...).detect(...).
    labels - array. Integers labelling each keypoint.
    
    Output:
    None. Img+keypoints are plotted.
    """
    #get coordinates and sizes of all keypoints
    x,y,sizes = getXYSfromKPS(kps)

    #plot background image
    cmap = plt.get_cmap(colormap,len(set(labels)))
    fig =plt.figure(figsize=(7,7), dpi=80)
    ax = plt.gca()
    plt.imshow(img, cmap=get_correctGreyCmap("Grey_cmap"), vmin=0, vmax=255)
    
    for i in range(len(set(labels))):
        c = np.array(cmap(i))
        for k in range(np.size(kps)):
            if labels[k]==i:
                ax.add_patch(mpl.patches.Circle((x[k],y[k]), radius=sizes[k], color=c, **kwargs))


def plot_one_cluster(img, kps, labels, cluster_label, colormap="jet",**kwargs):
    """
    Plots on top of the image the keypoints of the requested label.
    
    Parameters:
    -----------
    Input:
    img - array. Input image.
    kps - tuple. Keypoints, obtained through cv2.xfeatures2d.SIFT_create(...).detect(...).
    labels - array. Integers labelling each keypoint.
    cluster_label - int. Specific cluster label.
    
    Output:
    None. img + keypoints are plotted.
    """
    #get coordinates and sizes of all keypoints
    x,y,sizes = getXYSfromKPS(kps)

    #plot background image
    cmap = plt.get_cmap(colormap,len(set(labels)))
    plt.imshow(img,cmap=get_correctGreyCmap("Grey_cmap"),vmin=0,vmax=255)
    ax = plt.gca()
    c = np.array(cmap(cluster_label))
    for k in range(np.size(kps)):
        if labels[k]==cluster_label:
            ax.add_patch(mpl.patches.Circle((x[k],y[k]), radius=sizes[k], color=c, **kwargs))

            
def kNearestNeighbours(x, k, eps=0):
    '''
    Find and store the difference vectors of the k-nearest-neighbors for all the x's elements,
    i.e. a keypoint coordinate.

    Parameters:
    -----------
    Input:
    x - array with shape (N,m). N is number of points and m is dimension (2 = x,y coord.).
        Contains the keypoint's coordinates.
    k - int. Number of nearest neighbors to calculate the difference vectors.

    Output:
    KNN - array with shape (len(x)*k,2). Distances of the k-nearest-neighbours from any x[i].
    '''
    N,m = np.shape(x)
    if k >= N:
        #check if k is bigger or equal to the number of points
        print("k too big for array of size {}".format(N))
        print("setting k to {}-1".format(N))
        k = N-1
    kNN = np.array([])
    for i in range(N):
        #for every x check k-NN that are nonzero    
        dist = np.linalg.norm(-x+x[i,:], axis=1)
        sort_i = np.argsort(dist)
        dist = np.take_along_axis(dist, sort_i, axis=0)
        Nzeros = np.size(dist[dist<=eps])
        for j in range(k):
            kNN = np.append(kNN, x[sort_i[Nzeros+j],:]-x[i,:])
    kNN = kNN.reshape((N*k, m))
    return kNN


def plot_clustered_NNVs(kNN_right, kNN_labels, kNN_red, a, b):
    '''
    Plots the distances distribution of any keypoint with its k-nearest-neighbors (kNN).
    
    Parameters:
    ----------
    Input:
    kNN_right - array with shape (N,m). N is number of points and m is dimension (2 = x,y coord.).
                Contains the distance distributions, mirrored to have positive horizontal component. 
    kNN_labels - array. Contains the label of each computed NN distance.
    kNN_red - array with shape (k, 2). Average coordinates of the k-nearest-neighbors.
    a, b - arrays. Components of the found lattice vectors.
        
    Output:
    None. Makes the requested plot.
    '''
    plt.axis("equal")
    plt.gca().invert_yaxis()
    for i in set(kNN_labels):
        plt.scatter(kNN_right[kNN_labels == i,0],kNN_right[kNN_labels == i,1],s = 2,label=str(i))
    plt.scatter(kNN_red[:,0],kNN_red[:,1],label = "reduced",s = 4)
    plt.arrow(0,0,a[0],a[1])
    plt.arrow(0,0,b[0],b[1])

    
def DensityClustering(x, rtol, mins=3):
    '''
    Use DBSCAN algorithm to cluster a (potentially noisy) list of vectors.

    Parameters:
    ----------
    Input:
    x - array. Keypoint's coordinates.
    rtol - float. Distance measure used for the DBSCAN clustering.
    mins - integer. Minimum amount of samples to form cluster in DBSCAN.

    Output:
    unique - array with shape (k, 2). Average coordinates of the k-nearest-neighbors.
    labels - array. Contains the label of each computed NN distance.
    '''
    N, m = np.shape(x)
    clustering = DBSCAN(eps = rtol, min_samples = mins).fit(x)
    labels = clustering.labels_

    unique = np.array([])
    for i in set(labels):
        if i != -1:
            unique = np.append(unique, np.average(x[labels==i,:], axis=0))
    unique = unique.reshape((np.size(unique)//m,m))
    
    return unique, labels


###########START OF GRIDPLOT FUNCTIONS########################
##############################################################

def draw_line(start, a, x_lim = [0,1], y_lim=[0,1], **kwargs):
    '''
    Tries to draw a line in a box defined by x_lim and y_lim,
    returns 1 if it doesn't succeed.
    Used in utilities.py only.
    '''
    start, a = np.asarray(start), np.asarray(a)
    x = []
    # for all intersections with the x/y axis:
    # append intersection if y/x coordinate lies in boundaries
    if a[0]!=0:
        #intersection with left boundary at:
        t = (x_lim[0]-start[0]) / a[0]
        p = start+t*a
        if (p[1] >= y_lim[0]) & (p[1] <= y_lim[1]):
            x.append(p)
        #intersection with right boundary at:
        t = (x_lim[1]-start[0]) / a[0]
        p = start + t*a
        if (p[1] >= y_lim[0]) & (p[1] <= y_lim[1]):
            x.append(p)
    if a[1]!=0:
        #intersection with bottom boundary at:
        t = (y_lim[0]-start[1]) / a[1]
        p = start + t*a
        if (p[0] >= x_lim[0]) & (p[0] <= x_lim[1]):
            x.append(p)
        #intersection with top boundary at t= :
        t = (y_lim[1]-start[1]) / a[1]
        p = start + t*a
        if (p[0] >= x_lim[0]) & (p[0] <= x_lim[1]):
            x.append(p)
    x = np.asarray(x)
    if np.shape(x)[0] != 0:
        x = np.unique(x,axis=0)
        plt.plot(x[:,0], x[:,1], "k--", **kwargs)
        return 0
    else:
        return 1

    
def draw_periodic_lines(starting_point, line_direction, separation_vector, x_lim = [0,1],
                        y_lim = [0,1], **kwargs):
    '''
    Used in utilities.py only.
    '''
    starting_point = np.asarray(starting_point)
    line_direction = np.asarray(line_direction)
    separation_vector = np.asarray(separation_vector)

    success = draw_line(starting_point, line_direction, x_lim=x_lim, y_lim=y_lim, **kwargs)
    new_origin = starting_point + separation_vector
    while success == 0:
        success = draw_line(new_origin, line_direction, x_lim=x_lim, y_lim=y_lim, **kwargs)
        new_origin = new_origin + separation_vector
    new_origin = starting_point - separation_vector
    success = 0
    while success == 0:
        success = draw_line(new_origin, line_direction, x_lim=x_lim, y_lim=y_lim, **kwargs)
        new_origin = new_origin - separation_vector

        
def plot_grid(start, a, b, **kwargs):
    '''
    Used in utilities.py only.
    '''
    plt.plot(start[0], start[1], 'bo', label="origin")
    x_lim = np.sort(plt.xlim())
    y_lim = np.sort(plt.ylim())
    draw_periodic_lines(start, a, b, x_lim=x_lim, y_lim=y_lim, **kwargs)
    draw_periodic_lines(start, b, a, x_lim=x_lim, y_lim=y_lim, **kwargs)
    plt.legend()

    
#### Old unused function
def create_configfile(configfilepath,
                        inputfile,                               # Input
                        cThr, sigma, nOctLayers,                 # SIFT
                        size_Threshold, edge_Threshold,          # Keypoint filtering
                        clustering_span_kp,                      # Keypoint Clustering
                        k1,                                      # Nearest Neighbours 1
                        cluster_span_kNN, clustersize_Threshold, # Nearest Neighbours Clustering
                        clustering_span_SUBL,                    # Sublattice lookup
                        k2, rtol_rel, arrow_width):              # Deviation plot
    '''
    Takes all the parameters that go into creating a complete
    experiment and creates a config file that can then be
    taken to recreate the experiment.
    '''
    conf = configparser.ConfigParser(allow_no_value = True)
    conf.add_section('Input')
    conf['Input'] = {'inputfile' : inputfile}

    conf.add_section('SIFT')
    conf['SIFT'] =                 {';SIFT parameters are well explained in:':None,
                                    '; https://docs.opencv.org/3.4.9/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html':None,
                                    'contrast_threshold'     : cThr,
                                    'sigma'                  : sigma,
                                    'nOctaveLayers'          : nOctLayers}
    conf.add_section('Keypoint filtering')
    conf['Keypoint filtering'] =   {';These are thresholds to filter out keypoints that could get in the way of the method':None,
                                    ';Both are in units of the median keypoint size':None,
                                    ';All keypoints that are bigger than median*size_Threshold are deleted':None,
                                    ';All keypoints that are closer than median*edge_Threshold to one border of the image are deleted':None,
                                    'size_Threshold'         : size_Threshold,
                                    'edge_Threshold'         : edge_Threshold}
    conf.add_section('Keypoint Clustering')
    conf['Keypoint Clustering'] =  {';Clusterings with n Clusters between lower and upper bound are evaluated wrt their silhouette score':None,
                                    ';The one with the maximal silhouette score is chosen for further processing':None,
                                    'nClusters_lower_bound'  : clustering_span_kp[0],
                                    'nClusters_upper_bound'  : clustering_span_kp[-1]+1}
    conf.add_section('Nearest Neighbours')
    conf['Nearest Neighbours'] =   {';number_of_NN should be self explanatory (NN stands for nearest neighbours':None,
                                    ';usually to get a good clustering of the NN distribution lower bound should be set to k1':None,
                                    ';and upper bound to k1*4':None,
                                    ';clustersize_Threshold is used to reduce impact of erroneous NN-vectors on the selection of the lattice vectors':None,
                                    ';in the final distribution only NN-clusters with members N>=clustersize_Threshold*N_max are considered':None,
                                    ';Usually clustersize_Threshold is set to 0.3':None,
                                    'number_of_NN'           : k1,
                                    'nClusters_lower_bound'  : cluster_span_kNN[0],
                                    'nClusters_upper_bound'  : cluster_span_kNN[-1]+1,
                                    'clustersize_Threshold'  : clustersize_Threshold}
    conf.add_section('Sublattice lookup')
    conf['Sublattice lookup'] =    {';Clustering parameters for the sublattice distribution':None,
                                    'nClusters_lower_bound'  : clustering_span_SUBL[0],
                                    'nClusters_upper_bound'  : clustering_span_SUBL[-1]+1}
    conf.add_section('Deviation plot')
    conf['Deviation plot'] =       {';For the final deviation plot nearest Neighbours are calculated wrt to all keypoints':None,
                                    ';not just for the (presumably regular) keypoint cluster with most number of keypoints':None,
                                    ';therefore it is usually better to chose k2 bigger than k1':None,
                                    ';All vectors that are within the relative_r-Tolerance of the lattice vectors are drawn':None,
                                    ';The arrow_width can also be specified here (see matplotlib.quiver() - width parameter)':None,
                                    ';It is in units of the plot width (usual value 0.005)':None,
                                    'number_of_NN'           : k2,
                                    'relative_r-Tolerance'   : rtol_rel,
                                    'arrow_width'            : arrow_width}
    with open(configfilepath,'w') as cfile:
        conf.write(cfile)
        cfile.close()
    
