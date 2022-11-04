'''
UTILITIES AND PLOTTING
---------
Functions used in the notebook.
---------
'''
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy as sp
import matplotlib as mpl
from matplotlib.colors import NoNorm
import configparser
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from PIL import Image, ImageDraw
import matplotlib.patches as patches


def find_best_clustering(descriptors, span, sklearn_clustering=AgglomerativeClustering, **kwargs):
    '''
    Finds the optimal number of clusters of the given descriptors
    with respect to the silhouette score.
    
    Parameters:
    -----------
    descriptors - array of size (n,d), for n features with d descriptors;
    span - a range of values where the optimal number of clusters should be.
    **kwargs - additional possible arguments for AgglomerativeClustering.
    
    Result:
    -------
    best_labels - for each descriptor returns a label specifying the described
                  features cluster.
    '''
    max_SC = 0 #max score
    labels = np.zeros(np.shape(descriptors)[0])
    for N in span:
        clustering = sklearn_clustering(n_clusters=N, **kwargs).fit(descriptors)
        labels = clustering.labels_
        sil_score = silhouette_score(descriptors, labels)
        if sil_score > max_SC:
            best_labels = labels #each keypoint has an integer specifying its cluster
            max_SC = sil_score
    print("Maximal silhouette score found with {} clusters:\nsil_score = {}"
          .format(max(set(best_labels))+1, max_SC))
    return best_labels


def dist_per(u, v, a, b):
    '''
    Calculates distances in a 1x1 unit cell with periodic boundary conditions.
    
    Parameters:
    ----------    
    Input:
    u, v - arrays. Two points which distance is to be calculated.
    a, b - arrays. Coordinates of the lattice vectors.
    
    Output:
    d - float. Distance.      
    '''
    deltax = np.abs(u - v)
    d = np.sqrt(\
            np.sum(\
             np.matmul(np.array([a,b]).T,
                    np.amin(\
                        np.array([deltax, 1-deltax])\
                        ,axis=0))**2\
            )\
        )
    return d


def COM(x):
    '''
    Calculates the average position of each cluster (Center Of Mass)
    following COM calculation by Linge Bai and David Breen:
    "Calculating Center of Mass in an Unbounded 2D environment".
    
    Parameters:
    ----------    
    Input:
    x - array. Coordinates of keypoints belonging to the same cluster.
    
    Output:
    COM - array. Cordinates of the center of mass.
    '''
    
    N,k = np.shape(x)
    COM = np.zeros(2)
    ri = 1/(2*np.pi)
    #first component:
    theta_i = x[:,0]*2*np.pi
    x1 = np.zeros((N,3))
    x1[:,0] = ri*np.cos(theta_i)
    x1[:,1] = x[:,1]
    x1[:,2] = ri*np.sin(theta_i)
    X_bar = np.average(x1, axis=0)
    if np.abs(X_bar[0]) >= 0.001 and np.abs(X_bar[2])>=0.001:
        theta = np.arctan2(-X_bar[2],-X_bar[0])+np.pi
    else:
        theta = 0
    COM[0] = ri*theta
    #second component:
    theta_i = x[:,1]/ri
    x1[:,0] = x[:,0]
    x1[:,1] = ri*np.cos(theta_i)
    x1[:,2] = ri*np.sin(theta_i)
    X_bar = np.average(x1,axis=0)
    if np.abs(X_bar[0]) >= 0.001 and np.abs(X_bar[2])>=0.001:
        theta = np.arctan2(-X_bar[2], -X_bar[1])+np.pi
    else:
        theta = 0
    COM[1] = ri*theta
    return COM

def Rot_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s), (s, c)))

def Apply_PBC(a_vec, b_vec, point):
    basis_matrix = np.array([[a_vec[0], b_vec[0]], [a_vec[1], b_vec[1]]])
    trf = np.linalg.solve(basis_matrix, point)
    trf[0] -= round(trf[0])
    trf[1] -= round(trf[1])
    return np.dot(basis_matrix, trf)

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
    fig = plt.figure(figsize=(7,7), dpi=80)
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
    plt.imshow(img, cmap=get_correctGreyCmap("Grey_cmap"), vmin=0, vmax=255)
    ax = plt.gca()
    c = np.array(cmap(cluster_label))
    for k in range(np.size(kps)):
        if labels[k]==cluster_label:
            ax.add_patch(mpl.patches.Circle((x[k],y[k]), radius=sizes[k], color=c, **kwargs))


def clusters_selector(kps, labels, chosen_labels): #not used anymore
    '''
    Allows to filter the keypoints based on the cluster label. Useful for visualising clusters 
    belonging to features of interest when combined with a plotting function.
    
    Parameters:
    -----------
    Input:
    kps - tuple. Keypoints, obtained through cv2.xfeatures2d.SIFT_create(...).detect(...).
    labels - array. Integers labelling each keypoint.
    chosen_labels - list. It contains the labels i want to select.
    
    Output:
    kps_filtered - tuple or array?. Filtered keypoints.
    labels_filtered - array. Filtered labels.
    '''
    kps_filtered = []
    labels_filtered = []
    for j in range(len(labels)):
        for i in chosen_labels:     
            if labels[j] == i:
                #del kps_filtered[j]
                kps_filtered.append(kps[j])
                labels_filtered.append(labels[j])
                
    kps_filtered = tuple(kps_filtered)
    labels_filtered = np.array(labels_filtered)
            
    return kps_filtered, labels_filtered   
            
            
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

    
def plot_lattice_deviations(x, a, b, subl_labels, k, rtol_rel, path,
                             arrow_width, img, colorsmin, colorsmax):
    '''
    As a final result plot the kNN-vectors from their respective keypoint,
    colored by how much they deviate from the found lattice vectors.
    It uses a "rtol_rel" threshold to decide which kNN's to plot.
    '''
    kNN = kNearestNeighbours(x, k)
    gray_cmap = get_correctGreyCmap("greyscale")
    
    plt.figure(figsize=(8,8), dpi=80)
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.title('NN to consider in the deviation plot')
    plt.scatter(kNN[:,0], kNN[:,1], s=1)
    ax = plt.gca()
    ax.add_patch(patches.Circle(a, radius=rtol_rel, fill=False, color=(0,0,0)))
    ax.add_patch(patches.Circle(b, radius=rtol_rel, fill=False, color=(0,0,0)))
    ax.add_patch(patches.Circle(-b, radius=rtol_rel, fill=False, color=(0,0,0)))
    ax.add_patch(patches.Circle(-a, radius=rtol_rel, fill=False, color=(0,0,0)))
    plt.arrow(0, 0, a[0], a[1])
    plt.arrow(0, 0, b[0], b[1])
    plt.arrow(0, 0, -a[0], -a[1])
    plt.arrow(0, 0, -b[0], -b[1])
    plt.show()
    
    diff = []
    # Consistent coloring of arrows across quiver plots:
    quivernorm = mpl.colors.Normalize(vmin=colorsmin, vmax=colorsmax, clip=True)
    for s in set(subl_labels):
        # I want to plot the NN-vectors for each sublattice
        # for that I need proper masks to seperate the sublattices out of x and kNN
        isinSubl = subl_labels == s
        kNNofSubl = isinSubl.repeat(k)
        
        print('Sublattice {}:'.format(s))
        fig = plt.figure(figsize=(4,4), dpi=160)
        plt.imshow(img, cmap=gray_cmap, norm=NoNorm())
        plt.axis("off")

        # Plot all NN-vectors that belong to the FIRST basis vector:
        is_a = (np.linalg.norm(kNN-a, axis=1) <= rtol_rel) 
        is_minus_a = (np.linalg.norm(kNN+a, axis=1) <= rtol_rel)
        to_draw = kNNofSubl & (is_a | is_minus_a)
        kp_inds1 = np.where(to_draw)[0]//k
        kps1 = np.take(x, kp_inds1, axis=0)
        
        diff = np.append(diff, kNN[is_a & kNNofSubl,:]-a)
        diff = np.append(diff, kNN[is_minus_a & kNNofSubl,:]+a)
        diff_magn = np.zeros(np.shape(kNN[to_draw,:])[0])
        
        for i in range(np.shape(kNN[to_draw,:])[0]):
            if (is_a[to_draw])[i]:
                diff_magn[i] = np.linalg.norm((kNN[to_draw,:])[i,:]-a)
            elif (is_minus_a[to_draw])[i]:
                diff_magn[i] = np.linalg.norm((kNN[to_draw,:])[i,:]+a)
        quiv1 = plt.quiver(kps1[:,0], kps1[:,1], kNN[to_draw,0], kNN[to_draw,1],
                           diff_magn, cmap = "plasma", scale_units="xy", angles="xy", scale=1,
                           width=arrow_width, norm=quivernorm)
        diff_magn_a = diff_magn # needed for the histograms

        # Plot all NN-vectors that belong to the SECOND basis vector:
        is_b = (np.linalg.norm(kNN-b, axis=1) <= rtol_rel)
        is_minus_b = (np.linalg.norm(kNN+b, axis=1) <= rtol_rel)
        to_draw = (is_b | is_minus_b) & kNNofSubl
        kp_inds1 = np.where(to_draw)[0]//k
        kps1 = np.take(x, kp_inds1, axis=0)
        
        diff = np.append(diff, kNN[is_b & kNNofSubl,:]-b)
        diff = np.append(diff, kNN[is_minus_b & kNNofSubl,:]+b)
        
        diff_magn = np.zeros(np.shape(kNN[to_draw,:])[0])
        for i in range(np.shape(kNN[to_draw,:])[0]):
            if (is_b[to_draw])[i]:
                diff_magn[i] = np.linalg.norm((kNN[to_draw,:])[i,:]-b)
            elif (is_minus_b[to_draw])[i]:
                diff_magn[i] = np.linalg.norm((kNN[to_draw,:])[i,:]+b)
        quiv1 = plt.quiver(kps1[:,0], kps1[:,1], kNN[to_draw,0], kNN[to_draw,1],
                           diff_magn, cmap="plasma", scale_units="xy", angles="xy", scale=1,
                          width=arrow_width, norm=quivernorm)
        ax2 = plt.gca()
        cax = fig.add_axes([ax2.get_position().x1+0.04, ax2.get_position().y0, 0.04,
                            ax2.get_position().height], label="colorbar")
        cbar = plt.colorbar(quiv1, cax=cax)
        cbar.set_label("Deviation from avg. lattice vectors [px]")
        #plt.title("Deviation of NN-vectors from average lattice vector\nsublattice {}".format(s))
        plt.savefig(path+"deviations_{}.svg".format(s))
        plt.show()
        
        # Histograms
        fig, ax = plt.subplots()
        plt.title("Bond deviation distributions, sublattice {}".format(s))
        ax.hist(diff_magn_a, rwidth = 0.2, bins = 8, label='a')
        ax.hist(diff_magn, rwidth = 0.2, bins = 8, label='b')
        plt.legend()
        plt.show()

    diff = diff.reshape(np.size(diff)//2, 2)

    plt.figure(figsize=(8,8), dpi=80)
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.title("Distribution of deviations from primitive lattice vectors")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.scatter(diff[:,0], diff[:,1], s=1)
    plt.savefig(path+"deviations.svg")
    plt.show()    
    
    
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


# Average cell-rrlated functions

def average_cell(img, a, b, x, subl_labels, center_atom_cluster, lengthener):
    '''
    Uses the original image, the unit vectors, and the chosen sublattice 
    and returns the mean and average cells.
    '''
    # With lengthener 1 only the unit cells are averaged and displayed.
    # With lengthener = 2 the average is done with twice as long unit vectors 
    # leading to the the surrounding area being included in the average.

    orig_img = img
    coordinates = x[subl_labels == center_atom_cluster]

    display_points = np.zeros((len(coordinates), 4, 2)).tolist()  # Only for plotting

    for i in range(len(coordinates)):
        kp_x = int(round(coordinates[i][0]))
        kp_y = int(round(coordinates[i][1]))

        display_points[i][0][0] = int(
            kp_x + round((a[0] + b[0]) * 0.5 * lengthener)
        )
        display_points[i][0][1] = int(
            kp_y + round((a[1] + b[1]) * 0.5 * lengthener)
        )
        display_points[i][2][0] = int(
            kp_x + round((-1 * a[0] - b[0]) * 0.5 * lengthener)
        )
        display_points[i][2][1] = int(
            kp_y + round((-1 * a[1] - b[1]) * 0.5 * lengthener)
        )
        display_points[i][1][0] = int(
            kp_x + round((-1 * a[0] + b[0]) * 0.5 * lengthener)
        )
        display_points[i][1][1] = int(
            kp_y + round((-1 * a[1] + b[1]) * 0.5 * lengthener)
        )
        display_points[i][3][0] = int(
            kp_x + round((a[0] - b[0]) * 0.5 * lengthener)
        )
        display_points[i][3][1] = int(
            kp_y + round((a[1] - b[1]) * 0.5 * lengthener)
        )

    ##### Sum unit cells
    #  Gives averaged view of lattice
    imgList = []
    first = True
    cropped_list = []

    height, width, channels = orig_img.shape
    for i in range(len(coordinates)):
        mask = np.zeros(orig_img.shape[0:2], dtype=np.uint8)
        points = np.array(display_points[i])
        good_cell = True
        for k in range(len(points)):
            if (
                points[k][0] < 0
                or points[k][0] > width - 1
                or points[k][1] < 0
                or points[k][1] > height - 1
            ):
                good_cell = False
        if good_cell:
            cv.drawContours(mask, [points], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = orig_img

            rect = cv.boundingRect(points)  # returns (x,y,w,h) of the rect
            cropped = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
            cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
            cropped_list.append(cropped)

            ## crate the white background of the same size of original image
            wbg = np.ones_like(orig_img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            # overlap the resulted cropped image on the white background
            dst = wbg + res

            pil_im = Image.fromarray(cropped)
            imgList.append(pil_im)

            temp = np.asarray(cropped)
            temp = temp.astype("uint32")
            if first:
                sumImage = temp
                first = False
            else:
                sumImage = sumImage + temp


    avgArray = sumImage / len(imgList)
    avgImg = Image.fromarray(avgArray.astype("uint8"))
    # avgImg.save("av_cropped.png")

    fig = plt.figure()
    plt.imshow(avgImg, cmap="inferno")
    plt.tick_params(axis='both', which='both', bottom= False, top= False, labelbottom= False,
                    right= False, left= False, labelleft= False)
    plt.show()

    cropped_list = np.array(cropped_list)
    median_cropped = np.empty_like(cropped)

    for i in range(len(cropped)):
        for k in range(len(cropped[0])):
            median_cropped[i][k] = np.median(cropped_list[:, i, k])

    median_cropped = median_cropped.astype(np.int32)
    medianImg = Image.fromarray(median_cropped.astype("uint8"))

    fig = plt.figure()
    plt.imshow(median_cropped, cmap="inferno")
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)
    plt.show()

    width, height = avgImg.size
    center = np.array([width/2, height/2])

    return avgImg, medianImg, width, height


def plot_symmetry_cell(avgImg, medianImg, width, height, a_vec, b_vec, symmetry_bool,
                       lattice_construct, path, atomtypes, center_atom_cluster, corner_at_atomtype,
                       corner_at_atomgroup_index):
    
    fig, ax = plt.subplots()
    # To plot we need to invert y-axis
    a_vec[1] = -a_vec[1]
    b_vec[1] = -b_vec[1]

    label = [["A", None, None, None], ["B", None, None, None], ["C", None, None, None]]
    colors = ["tab:blue", "tab:orange"]
    
    sublattice_index_label = 0
    for i in range(len(lattice_construct)):
        self_trf_Index = 0
        
        for k in range(len(lattice_construct[i])):
            plt.scatter(lattice_construct[i][k][0], -lattice_construct[i][k][1],
                        label=sublattice_index_label, marker="o", s=300, edgecolors='black',
                        zorder=20)
            sublattice_index_label += 1

    ax.legend()
    #offset = [-lattice_construct[0][0][0], lattice_construct[0][0][1]] #tmp_var, unused
    brillouin_zone_plot = np.array([a_vec*0, a_vec, a_vec + b_vec, b_vec, a_vec*0])
    - np.array(lattice_construct[corner_at_atomtype][corner_at_atomgroup_index])
    #- np.array([0.33*a_vec[0], 0.33*b_vec[1]])
    
    patch = patches.PathPatch(mpl.path.Path(brillouin_zone_plot), alpha=1, lw=5, zorder=5,
                              facecolor='none', edgecolor="ghostwhite")

    
    ax.add_patch(patch)

    plt.imshow(avgImg, cmap="inferno", extent=[-width/2., width/2., -height/2., height/2. ])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)

    plt.savefig(path+"symmetry_cell_avg.svg", bbox_inches='tight', dpi=200)

    plt.imshow(medianImg, cmap="inferno", extent=[-width/2., width/2., -height/2., height/2. ])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)

    plt.savefig(path+"symmetry_cell_median.svg", bbox_inches='tight', dpi=200)
    #a_vec[1] = -a_vec[1]
    #b_vec[1] = -b_vec[1]
    ############################################################


    
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
    
