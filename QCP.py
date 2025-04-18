# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import patheffects
import cv2 as cv
import PIL.Image
import PIL.ImageDraw
from scipy.cluster.hierarchy import fcluster
import configparser
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from sklearn.svm import SVC




#external utility functions
def angle(v1, v2):
    """Basic vector angle calculator
    
    This function calculates the angle between two vectors. Works with lists as well.

    Args:
        v1 (array): First vector.
        v2 (array): Second vector.

    Returns:
        float: Angle between the two given vectors in radians.

    """
    return np.arccos((v1@v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))


def get_pairwise_min_distance(x):
    """Pairwise minimum distance calculator
    
    Calculates the distance for each point in a small array of vectors to its nearest neighbour. Works with lists as well.

    Args:
        x (array): Array containing vectors. Array shape should be (m,n) where n is the vector dimension.

    Returns:
        pairwise_min (array): Array (of length m) containing the minimum distances to each point's nearest neighbour.

    """
    #convert to numpy array if x was given as a list
    x=np.array(x)
    
    #returns sorted distances between points, excluding distance to self.
    pairwise = np.sort(np.linalg.norm(x[:,np.newaxis] - x, axis=2),axis=1)[...,1:]
    
    #get array of minima, ensure it is a flat array
    pairwise_min = np.min(pairwise, axis=1).reshape(-1)
    
    return pairwise_min


def find_best_clustering(data, span, return_score=False, **kwargs):
    """Clustering score analyzer
    
    Finds the best number of clusters for the given data. The performance of the cluster amounts
    are determined by the silhouette score of their respective clustering results.
    

    Args:
        data (array): Data to be clustered. This should be an array or list of shape (m,n) where m is the number of data points
            and n is their dimension.
        span (range): Range from lower to upper bound of the desired number of clusters. Cannot be lower than 2. The optimum cluster amount
            is selected from this.
        return_score (bool, optional): Toggle option for returning the score of the clustering or not. Defaults to False.
        **kwargs : Additional arguments for sklearn.cluster.AgglomerativeClustering.

    Returns:
        labels (array):  Labels for each data point. Each label is an integer.
        score (float): Only returned if 'return_score' is True.
        
        If no score is to be returned, a simple array is returned instead of a tuple.

    """
    
    #initialize some values to be altered later
    max_sc = 0  #maximum silhouette score
    labels = np.zeros(np.size(data)) #empty label list
    best_labels = labels
    
    #create hierarchical clustering Tree
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, **kwargs).fit(data)
    
    # create the counts of samples under each node
    # source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, 
                                      counts]).astype(float)
    
    
    #iterate through the cluster amounts in the set span and cut off tree at desired level
    for n in span:
        
        #get labels
        labels = fcluster(linkage_matrix,t=n,criterion='maxclust')-1
        
        #calculate silhouette score
        sil_score = silhouette_score(data, labels)
        
        #check if it is the maximum
        if sil_score > max_sc:
            best_labels = labels 
            max_sc = sil_score
    
    #return results
    if return_score:
        return best_labels, max_sc
    else:
        return best_labels


def get_kNNs(x, k, leafsize=20):
    """Nearest neighbour finder
    
    Finds the k nearest neighbours for each item in x using the KDtree algorithm.

    Args:
        x (array): Data points. Array must be of shape (m,n) where m is the number of points 
            and n is their dimension.
        k (int): Amount of nearest neighbours to find.
        leafsize (int, optional): leaf_size parameter for sklearn.neighbours.KDtree, 
            used to optimize performance. Defaults to 40.

    Returns:
        loc (array): Locations of the k nearest neighbours of each point.
        dist (array): Distances to the k nearest neighbours of each point.

    """
  
    
    #create tree object and apply it to data
    tree = KDTree(x, leaf_size = leafsize)
    dist, ind = tree.query(x, k=k+1)
    
    #remove nearest neighbour, which is the point itself
    dist = dist[:,1:]
    ind = ind[:,1:]
    
    #get locations
    loc = x[ind]
    
    return loc, dist 


# class setup
class QCP:
    """ 
    A class for pattern recognition, to be used on atomic resolution images or quasiperiodic tiling vertex point rendition.
    
    """
    
    
    ###
    # initialization
    ###
    
    def __init__(self, c_thr = 0.003, e_thr = 10., sigma=4., n_oct_layers = 8, 
                 size_threshold = 2., edge_threshold = 1., 
                 shrink_factor = 0.66, kNN_amount = 7,
                 path = None, filename = None, print_parameters=False):
        
        """Default class initialization
        
        Initializes the QuasiCrystal Pattern extractor (QCP) class instance. 
        If no 'path' or 'filename' is given, the other arguments 
        are used to set class attributes (of the same name) crucial to tuning the pattern recognition. 
        Setting a 'path' and 'filename' overrides any arguments using the ones given in the .ini file and calls the 
        class function 'set_image'. 
        

        Args:
            c_thr (float, optional): Contrast threshold for keypoint detection. Defaults to 0.003.
            e_thr (float, optional): Edge threshold for keypoint detection. Defaults to 10..
            sigma (float, optional): Blur factor applied at each octave. Defaults to 4..
            n_oct_layers (int, optional): Number of layers per octave. Defaults to 8.
            size_threshold (float, optional): Size parameter for discarding keypoints
                Used to filter out very big or very small keypoints. Defaults to 2..
            edge_threshold (float, optional): Edge parameter for discarding keypoints. 
                Used to filter out keypoints close to the edge of the image. Defaults to 1..
            shrink_factor (float, optional): Multiplication <1. by which to shrink keypoint size. 
                Helps with clustering keypoints by central luminosity. Defaults to 0.66.
            kNN_amount (int, optional): Amount of nearest neighbours to be found for each keypoint. Defaults to 7.
            path (str, optional): System path to image and parameters.ini file. Defaults to None.
            filename (str, optional): File name of the image to be used. Defaults to None.
            print_parameters (bool,optional): Adds option to print the parameters used in the instance. Defaults to False.
        """
        
        #set parameter defaults
        #SIFT parameters
        self.c_thr = c_thr
        self.e_thr = e_thr
        self.sigma = sigma
        self.n_oct_layers = n_oct_layers
        
        #keypoint filtering parameters
        self.size_threshold = size_threshold
        self.edge_threshold = edge_threshold
        self.shrink_factor = shrink_factor
        
        #nearest neighbour parameter
        self.kNN_amount = kNN_amount
        
        #override parameters if there is a parameters.ini file in the specified path
        if path != None and os.path.isfile(path+"parameters.ini"):
            
            conf = configparser.ConfigParser(allow_no_value=True)
            conf.read_file(open(path+"parameters.ini"))
            
            #[SIFT]
            self.c_thr = conf.getfloat('SIFT', 'c_thr', fallback=0.003)
            self.e_thr = conf.getfloat('SIFT', 'e_thr', fallback=10)
            self.sigma = conf.getfloat('SIFT', 'sigma', fallback=4)
            self.n_oct_layers = conf.getint('SIFT', 'n_oct_layers', fallback=8)
            
            #[Keypoint filtering]
            self.size_threshold = conf.getfloat('Keypoint filtering', 'size_threshold', fallback=2)
            self.edge_threshold = conf.getfloat('Keypoint filtering', 'edge_threshold', fallback=1)
            self.shrink_factor = conf.getfloat('Keypoint filtering', 'shrink_factor', fallback=0.66)
            
            #[Nearest Neighbours]
            self.kNN_amount = conf.getint('Nearest Neighbours', 'kNN_amount', fallback=7)
        
        #set up image
        if path!= None and filename != None:
            
            self.set_image(path+filename)
        
        #initialize empty keypoints list
        self.kp = np.array([])
        
        #perform printing if so whished
        if print_parameters:
            print("SIFT contrast threshold:", self.c_thr)
            print("SIFT edge threshold:", self.e_thr)
            print("SIFT Gaussian sigma:", self.sigma)
            print("SIFT Number of Layers per octave", self.n_oct_layers)
            print("Filter size threshold:", self.size_threshold)
            print("Fitler edge threshold:", self.edge_threshold)
            print("Keypoint shrink factor:", self.shrink_factor)
            print("Nearest neighbours to analyze:", self.kNN_amount)
        
    ###
    # utility functions
    ###
    
    def set_image(self, inputfile):
        """Image Setter
        
        Sets the image to be analyzed. An original and a grayscale version are saved.

        Args:
            inputfile (str): Filename of image or path to image. The former requires script to be executed in the same folder as the file.

        Returns:
            None.

        """
        self.img = cv.imread(inputfile)
        self.gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
    
        
    def calculate_filtered_keypoints(self):
        """
        Keypoint detection, filtering and descriptor calculation
        
        Creates a cv.SIFT instance that is used to detect keypoints in an image. The keypoints are then filtered
        based on threshold set in the class initialization in the following way:
        - remove keypoints whose octave is below 'min_layer' or above 'max_layer'
        - remove keypoints whose size is above 'size_threshold' multiplied with the median keypoint size or
        below the median size divided by 'size_threshold'.
        - remove keypoints whose distance to the image border is smaller than 'edge_threshold' multiplied with
        the median keypoint size.
        - remove duplicates of keypoints.
        The remaining keypoints are then resized to all be of size 'shrink_factor' multiplied with the new median keypoint size
        Using these keypoints, the associated descriptors are then calculated. Both keypoints and descriptors are 
        saved as internal class attributes for further use. See class attributes for further information.
        

        Returns:
            None.

        """
        
        #initialize sift instance
        self.sift = cv.SIFT.create(contrastThreshold=self.c_thr, edgeThreshold = self.e_thr, 
                              sigma=self.sigma, nOctaveLayers=self.n_oct_layers)
        
        #detect keypoints
        self.kp = np.concatenate((self.kp,self.sift.detect(self.gray, None)))     
        
        #remove wrongly sized keypoints
        sizes = [self.kp[i].size for i in range(np.size(self.kp))]
        size_med = np.median(sizes)
        wrongsize_index = [i for i in range(np.size(self.kp))\
                            if self.kp[i].size>self.size_threshold*size_med or 
                            self.kp[i].size<size_med/self.size_threshold]
        self.kp = np.delete(self.kp, wrongsize_index)
        
        #remove keypoints too close to image border, as positions may be wrong
        border = self.edge_threshold*size_med
        edge_kp_indices = []
        for i in range(np.size(self.kp)):
            if (self.kp[i].pt[0]<border or self. kp[i].pt[0]>np.shape(self.gray)[1]-border or\
                self.kp[i].pt[1]<border or self.kp[i].pt[1]>np.shape(self.gray)[0]-border):
                edge_kp_indices.append(i)        
        self.kp = np.delete(self.kp, edge_kp_indices)
        
        #remove multiples of keypoints by analyzing locations
        locs = np.array([self.kp[i].pt for i in range(np.size(self.kp))])
        locs, unique_loc_indices = np.unique(locs, return_index = True, axis = 0)
        self.kp = self.kp[unique_loc_indices]
        
        #get new median size
        size_med = np.median([self.kp[i].size for i in range(np.size(self.kp))])
        
        #resize keypoints to all be a portion of the median size
        for i in range(np.size(self.kp)):
            self.kp[i].size = size_med*self.shrink_factor
            self.kp[i].angle = 0
        
        #get final size
        self.median_kp_size = size_med*self.shrink_factor
        
        
        #compute descriptors
        self.kp, self.des = self.sift.compute(self.gray,self.kp)

        #convert keypoints from tuple to numpy array
        self.kp = np.array(self.kp)
    
    
    def calculate_kp_clusters(self, all_manual=False,**kwargs):
        """Keypoint cluster calculation
        
        Applies the global find_best_clustering function to the descriptors to get the keypoint labels.
        In the case that all keypoints were set manually from a list of coordinates it is assumed that all of
        them are useful and therefore there is only one cluster. 
        
        Args:
            all_manual (bool, optional): Indicates whether keypoints were detected automatically or added manually.
                Defaults to False.
            **kwargs: Additional arguments for find_best_clustering.

        Sets:
            kp_labels (array): Cluster labels of the keypoints given as an array of integers.

        """
        if all_manual:
            self.kp_labels = [0 for kp in self.kp]
        else: 
            self.kp_labels = find_best_clustering(self.des, [2], **kwargs)
        
    
    def get_informative_label_neighbours(self, **kwargs):
        """Relevant neighbour finder
        
        Finds which label containts the most useful intormation for the tiling
        from nearest neighbour distance clustering. The keypoint cluster whose nearest neighbour analysis
        shows the most structure (determined by the maximum silhouette score) is chosen. 
        Class attribute 'kNN_amount' dictates the k nearest neighbours to be found.

        Args:
            **kwargs: Additional arguments for find_best_clustering.

        Sets:
            kp_coord (array): Coordinates of all the keypoints.Can be used for display purposes.
            kp_label (int): Label of keypoint cluster containing information.
            locations_NN (array): Coordinates of k nearest neighbours.
            distances_NN (array): Distances to k nearest neighbours.
            distance_labels (array): Labels of the distance clusters of the k nearest neighbours.

        """
        
        #get keypoint coordinates
        self.kp_coord = np.array([self.kp[i].pt for i in range(np.size(self.kp))])

        #get unique labels
        unique_labels = np.unique(self.kp_labels)

        #initialize empty lists for holding location and distance information
        loc = [[] for i in unique_labels]           #NN locations
        dist = [[] for i in unique_labels]          #distances to NN
        dist_score = [[] for i in unique_labels]    #silhouette score for best distance clustering
        dist_labels = [[] for i in unique_labels]   #clustering labels 
        
        #loop through labels and perform nearest neighbour analysis as well as distance clustering
        for l, label in enumerate(unique_labels):
            
            #filter keypoint coordinates by label
            label_kp_coord = self.kp_coord[np.where(self.kp_labels==label)[0]]
            
            #get locations of and distances to NN
            loc[l], dist[l] = get_kNNs(label_kp_coord, self.kNN_amount)
            
            #cluster distances
            dist_labels[l], dist_score[l] = find_best_clustering(dist[l].reshape((-1,1)),
                                                                 range(2,self.kNN_amount), return_score=True,
                                                                 **kwargs)
           
        #if there is more structure in one label, the clustering score will be greater
        max_score_index = np.argmax(dist_score)
        
        #get the final arrays for the output
        self.kp_label = unique_labels[max_score_index]
        self.locations_NN = loc[max_score_index]
        self.distances_NN = dist[max_score_index]
        self.distance_labels = dist_labels[max_score_index]
    
    
    def get_classifier(self,**kwargs):
        """SVM training
        
        Trains a support vector machine on the descriptors to differentiate on future keypoint candidates.

        Args:
            **kwargs : Additional arguments for sklearn.svm.SVC

        Sets: 
            classifier (sklearn.svm.SVC): trained support vector machine
        
        Returns:
            None.

        """
        
        #create label training data using booleans, such that the "right" label returns a "True" prediction
        bool_train_labels = np.zeros_like(self.kp_labels, dtype=bool)
        bool_train_labels[self.kp_labels==self.kp_label]=True
        
        #train the support vector classifier
        self.classifier=SVC(**kwargs).fit(self.des, bool_train_labels)
    
    def center_locations(self):
        """Keypoint coordinate centering
        
        Centers each keypoint at (0,0) together with its k nearest neighbours. Needed for accurate connection vectors.
        
        Sets:
            label_kp_indices (array): indices of the keypoints belonging to the "chosen" cluster
            label_kp_coord (array): locations of each keypoint in label_kp_indices
            label_kp_coord_3d (array): label_kp_coord reshaped so that it has the same dimensions as locations_NN
            locations_NN_centered (array): the locations of the nearest neighbours, centered around (0,0)
        
        Returns:
            None.
        """
        
        self.label_kp_indices = np.where(self.kp_labels==self.kp_label)[0]
        self.label_kp_coord = self.kp_coord[self.label_kp_indices]
        self.label_kp_coord_3d = np.reshape(np.repeat(self.label_kp_coord, self.kNN_amount, axis=0), 
                                       (len(self.label_kp_coord),self.kNN_amount,2))
        self.locations_NN_centered = self.locations_NN - self.label_kp_coord_3d
    
    
    def calculate_dominant_distance_vectors(self, verbose = False, **kwargs):
        """Vector calculation
        
        Nearest neighbour locations are used to find the symmetry vectors through clustering. Clustering for symmetry number
        is limited to 8, 10 or 12, in accord with the nature of quasicrystals.
        
        Args:
            verbose (bool): Choice if the rotational symmetry findings should be reported. Defaults to True.
            **kwargs : Additional arguments for find_best_clustering.
        
        Sets:
            distance_label (int): Cluster number indicating where the next atoms in symmetry are.
            vectors (array): Vectors connecting the next atom positional clusters to origin (0,0).
            vector_length (float): median distance to the distance_label cluster. This is the length of the vectors.
            
        Returns:
            None.

        """
        
        #reshape locations to have the same first axis size as distance labels
        locations = np.copy(self.locations_NN_centered).reshape((-1,2))
        
        #find which label is the most prevalent and get locations only for that label, as well as distance
        unique, counts = np.unique(self.distance_labels, return_counts=True)
        self.distance_label = unique[np.argmax(counts)]
        where_distance_highest_count = np.where(self.distance_labels == self.distance_label)[0] 
        locations_for_clustering = locations[where_distance_highest_count]
        length = np.median(self.distances_NN.ravel()[where_distance_highest_count])
        
        
        #find 8- 10- or 12-fold rotational symmetry vectors (halved)
        labels = find_best_clustering(locations_for_clustering, [8,10,12], **kwargs)
        
        #print which kind of rotational symmetry has been found
        if verbose:
            print("Found {}-fold rotational symmetry".format(len(np.unique(labels))))
        
        #get the median locations of the nearest neighbour clusters
        median_locations = []
        for label in np.unique(labels):
            label_index = np.where(labels==label)
            median_locations.append(np.median(locations_for_clustering[label_index], axis=0))
            
        #set remaining attributes
        self.vectors = np.array(median_locations)
        self.vector_length = length

        
    def set_contours(self):
        """Contour setter
        
        Draws lines connecting the atom by their symmetry vectors as a binary image and extracts each cell/tile
        as a "contour" together with its hierarchy information. Uses cv.findContours for this.
        
        Sets:
            connections (array): Binary image with 0 everywhere except the atom conenctions of width 1.
            contours (array): Individual contours in the binary image, given as an array of all points belonging to each contour (i.e. the connecting lines).
            hierarchy (array): Information about parent and child for each contour.
    
        Returns:
            None.

        """
        
        #set an array same as original image size to all zero
        black = np.zeros_like(self.gray)
        
        #reshape the reference point and nearest neighbour locations to be an array of shape (n, 2)
        label_kp_reshaped = np.copy(self.label_kp_coord_3d).reshape((-1,2))
        locations_reshaped = np.copy(self.locations_NN).reshape((-1,2))
        
        #draw lines in the array connecting the (next rounded integers to) points of interest
        for i in np.where(self.distance_labels==self.distance_label)[0]:
            
            #set start and end of line
            line_start = np.rint([label_kp_reshaped[i][0],label_kp_reshaped[i][1]]).astype(np.int32)
            line_end = np.rint([locations_reshaped[i][0],locations_reshaped[i][1]]).astype(np.int32)
            
            #draw on zero array lines with thickness 1
            black = cv.line(black, line_start, line_end, (255,255,255), 1)
        
        #save the array containing all the connections between atoms
        self.connections = black
        
        #find the contours in the array.
        self.contours, self.hierarchy = cv.findContours(self.connections, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        
    def match_keypoints_to_contours(self, region = 6., predicted_shape_corners=[3,4,6]):
        """Keypoint assigner
        
        Finds which keypoints are associated with which contour i.e. finds which atoms span which tile. 
        Also starts to filter out which contours are definitely not tiles (based on number of corners).
        
        Args:
            region (float, optional): Radius around points in the contour where a keypoint should be to be associated. 
                Must be smaller than the minimum keypoint distance to work correctly. Excessively 
                small values can lead to failed associations. Defaults to 6..
            predicted_shape_corners (list, optional): Each list entry is the amount of corners (associated with keypoints) in the contour
                If an idea of the shape of the tiles exists, then the amount of 
                corners in the shapes should be passed. E.g. only triangles -> [3].
                If unsure, start with default. The only (small) difference lies in computation time. Defaults to [3,4,6].
        
        Sets:
            kp_tile_id (list): List of which keypoints belong to tile candidate contours.
            closed_contours (list): List of contours that could represent a tile.
            kp_open_id (list): List of which keypoints belong to contours that are flagged as non-tiles
            open_contours (ist): List of which contours do not represent a tile.
        
        Returns:
            None.

        """
        #list for holding "valid" tiles along with their matched contours
        self.kp_tile_id = []
        self.closed_contours = []
        
        #list for holding "invalid" tiles, which could be "open" regions where undetected points lie
        #second list contains the contours, used for better point backfilling later on
        self.kp_open_id = []
        self.open_contours = []
        
        #create tree object and apply it to data
        contour_tree = KDTree(self.label_kp_coord, leaf_size = 20)
        
        for c, contour in enumerate(self.contours):
            
            #reshape to (n,2) array
            contour = contour.reshape(-1,2)
            
            #only do process if we are dealing with "inner" contours
            if self.hierarchy[0][c][3]!=-1:
                
                kp_ids=[]
                
                #get the distances and indices of the nearest neighbour for each point in the contour
                dists, inds=contour_tree.query(contour, k=1)
                
                #only append the index of the keypoint if the distance is within the acceptable region. 
                for i in range(len(dists)):
                    if dists[i]<region:
                        kp_ids.append(inds[i])
                    
                    
                #reshape keypoint ids to unique-valued numpy array
                #ensures multiples of a keypoint close to contour points don't happen
                kp_ids = np.unique(np.array(kp_ids).reshape(-1))
                
                #if the number of keypoints associated is in the predicted corner list, tile is "valid"
                if len(kp_ids) in predicted_shape_corners:
                    self.kp_tile_id.append(kp_ids)
                    self.closed_contours.append(contour)
                
                #other contour results go into the open regions list
                elif len(kp_ids)>2:
                    self.kp_open_id.append(kp_ids)
                    self.open_contours.append(contour)
                
                
    def get_tile_angles(self, predicted_shape_corners=[3,4,6]):
        """Tile angle calculator
        
        Using keypoints and nearest neighbours finds the corners of the tiles, so that they can be later classified as shapes.
        Filters out some more tile candidates as well.

        Args:
            predicted_shape_corners (list, optional): Each list entry is the amount of corners (associated with keypoints) in the contour
                If an idea of the shape of the tiles exists, then the amount of 
                corners in the shapes should be passed. E.g. only triangles -> [3].
                If unsure, start with default. The only (small) difference lies in computation time. Defaults to [3,4,6].
                
        Sets:
            tile_angles (array): Internal corner angles associated with each tile. Each array entry has the length 
                max(predicted_shape_corners).

        Returns:
            None.

        """
        
        #get maximum and minimum distance
        correct_distances = self.distances_NN.reshape(-1)[np.where(self.distance_labels
                                                                   ==self.distance_label)[0]]
        min_distance = np.min(correct_distances)
        max_distance = np.max(correct_distances)
        
        
        #initialize list to hold all tile angles
        tile_angles = []
        
        #initialize list of "bad" indices. these happen when connections overlap, forming extra contours
        #that do not conform to the distance distribution between two vertices
        bad_indices = []
        
        #loop through the "good" tiles
        for t, tile in enumerate(self.kp_tile_id):
            
            #get locations of corner points
            point_locations = self.label_kp_coord[tile]
            
            #get the vectors between points
            point_vectors = point_locations[:,np.newaxis]-point_locations
            
            #get distances associated
            point_distances = np.linalg.norm(point_vectors,axis=2)
            
            #initialize array for angles of each tile
            angles = []
            
            #loop through vector groups
            for v, vec_group in enumerate(point_vectors):
                
                #for triangles only remove vector of point to itself
                if len(vec_group)==3:
                    vec_pair=vec_group[np.where(point_distances[v]!=0)[0]]
                
                #for shapes with >3 corners find which vectors correspond to the symmetry distances
                else:
                    vec_pair = vec_group[np.where((point_distances[v]>=min_distance) 
                                              & (point_distances[v]<=max_distance))[0]]
                
                
                if len(vec_pair)!=2:
                    bad_indices.append(t)
                    break
                    
                #get angle between the found vectors
                pair_angle = angle(vec_pair[0],vec_pair[1])
                
                #append to angle list for current tile
                angles.append(np.rad2deg(pair_angle))
            
            #bring angle arrays to the same shape as the maximum amount of angles a shape can have
            if len(angles)<np.max(predicted_shape_corners):
                angles = np.append(np.zeros(np.max(predicted_shape_corners)-len(angles)),angles)
            
            #convert to numpy array for consistency and sort for easier clustering
            angles = np.sort(np.array(angles).reshape(-1))
            
            #append to final list
            tile_angles.append(angles)
        
        #create class attribute 
        self.tile_angles = tile_angles
        
        #move entries that do not conform to "open" tiles
        for b_i in bad_indices[::-1]:
            self.kp_open_id.append(self.kp_tile_id[b_i])
            self.open_contours.append(self.closed_contours[b_i])
            del self.kp_tile_id[b_i]
            del self.closed_contours[b_i]
            del self.tile_angles[b_i]
    
    
    def filter_outlier_shapes(self, max_deviation=4.):
        """Shape filter
        
        Removes the tiles that are not similar to any other tile within a certain range. Will always filter out some tiles,
        so should be applied carefully. If a lot of tiles are wrongfully filtered out, try to leave this method out and
        see if the result is better. (The better the keypoint detection works, the less likely it is that this method
        is needed.)
        

        Args:
            max_deviation (float, optional): Multiplicator of the standard deviation to set the accepted range. 
            A value of 4 is 99.9% of the population assuming gaussian distribution i.e. only 0.01% of the distribution
            range will be labeled as "outlier". Defaults to 4..

        Returns:
            None.

        """
        
        #get minima of the pairwise distances
        pairwise_min = get_pairwise_min_distance(self.tile_angles)
            
        #get standard deviation
        pairwise_min_std = np.std(pairwise_min)
        
        #get indices of outlying tiles
        outlier_indices = np.where(pairwise_min>max_deviation * pairwise_min_std)[0]
        
        #move the tiles with the found indices (and update tile angles as well)
        for o_i in outlier_indices[::-1]:
            self.kp_open_id.append(self.kp_tile_id[o_i])
            self.open_contours.append(self.closed_contours[o_i])
            del self.kp_tile_id[o_i]
            del self.closed_contours[o_i]
            del self.tile_angles[o_i]
    
    
    def cluster_shapes(self, shape_span=range(2,7), max_clusters=3):
        """Tile group identifier
        
        Finds the clusters of shapes found in the tiles, essentially differentiating between triangles, rhombi, squares, etc.
        Filters out some more tiles that do not belong to the 3 (or max_clusters) dominant shape clusters.
        

        Args:
            shape_span (range, optional): Range of the amounts of clusters to be found. Defaults to range(2,7).
            max_clusters (TYPE, optional): To get rid of possible non-valid tiles (stemming from e.g. wrong detection, no detection), 
                pick a maximum amount of clusters. Quasicrystals with more than 3 different tiles making up the pattern
                have not yet been found. Defaults to 3.
        
        Sets:
            shape_labels (array): Labels for the tiles.
        
        Returns:
            None.

        """
        
        #find labels of the best-fitting clustering
        self.shape_labels = find_best_clustering(self.tile_angles, shape_span)
        
        #get the unique labels and counts
        unique_labels, counts = np.unique(self.shape_labels,return_counts=True)

        #if too many clusters are found, get which labels do not belong to 
        #the n specified clusters with most members
        if len(unique_labels)>max_clusters:
            wrong_labels = unique_labels[np.argsort(counts)[::-1]][max_clusters:]
            
            #get all the indices of the wrong labels
            wrong_indices = np.array([])
            for wrong_label in wrong_labels:
                wrong_indices = np.hstack((wrong_indices, np.where(self.shape_labels==wrong_label)[0]))
            
            #sort array to be able to loop through it
            wrong_indices.sort()
            
            #loop through from the back and "move" wrong tiles 
            for w_i in wrong_indices[::-1]:
                w_i = int(w_i)
                self.kp_open_id.append(self.kp_tile_id[w_i])
                self.open_contours.append(self.closed_contours[w_i])
                del self.kp_tile_id[w_i]
                del self.closed_contours[w_i]
                del self.tile_angles[w_i]
            
            #cluster again with the max clusters
            self.shape_labels = find_best_clustering(self.tile_angles, span=[max_clusters])
    
    
    def get_average_angles(self):
        """Tile group average calculation
        
        Finds the average tile shape of each shape cluster (so that the individual building blocks of the pattern can be displayed.)
        
        Sets:
            average_angles (list): List holding the average corner angles of each shape cluster.
        
        Returns:
            None.

        """

        #get the unique shape labels
        unique_shape_labels = np.unique(self.shape_labels)
        
        #initialize list to hold the average angle values 
        self.average_angles = [[] for u in unique_shape_labels]
        
        #loop through each shape and get the median angle
        for i, u in enumerate(unique_shape_labels):
            
            #get all the shapes belonging to the chosen label
            label_shapes = np.array(self.tile_angles)[np.where(self.shape_labels == u)[0]]
            
            #get the median values
            angle_median = np.median(label_shapes, axis=0)
            
            #add to list
            self.average_angles[i] = angle_median       
    
    
    def fill_tiles(self,colors = [(255,0,0),(0,255,0),(0,0,255)]):
        """Tile filling
        
        Fills the contours in the binary image that have been identified as tiles with colors, depending on which
        shape cluster the tile belongs too. Will be used to find missed points in open areas as well as potential 
        pattern result display.
        

        Args:
            colors (list, optional): List containing three tuples of shape 3, giving the RGB values that should be 
                associated with each shape cluster. Only relevant for display purposes. Defaults to [(255,0,0),(0,255,0),(0,0,255)].
        
        Sets:
            rgb_tile_lines (array): RGB image containing the color-filled tiles of the original contours binary image
            filled_tiles_bin (array): Binary image where areas that could potentially contain more atoms are 0 and the rest is 1.
        
        Returns:
            None.

        """
        
        #convert the binary contour image to RGB
        rgb_tile_lines = cv.cvtColor(self.connections,cv.COLOR_GRAY2RGB)
        
        #convert again to image object
        rgb_tile_lines = PIL.Image.fromarray(rgb_tile_lines)
        
        #get the midpoint of each valid tile
        tile_midpoints = np.array([np.mean(self.label_kp_coord[kp_tile],axis=0) 
                                    for kp_tile in self.kp_tile_id]).astype(np.int32)
        
        #draw unto image:
        for t, tile in enumerate(tile_midpoints):   
            PIL.ImageDraw.floodfill(rgb_tile_lines, tuple(tile), colors[self.shape_labels[t]])
        
        #convert back to array and save as class attribute
        self.rgb_tile_lines = np.asarray(rgb_tile_lines)
        
        #save binary copy to help with filling in missing tiles
        self.filled_tiles_bin = np.clip(cv.cvtColor(self.rgb_tile_lines, cv.COLOR_BGR2GRAY),0,1)
        
        
               
    def find_missing_points(self, acceptance_threshold=1., border_distance=2/3, decision_boundary=0.0):
        """Missing point finder
        
        This applies an algorithm that scans the open areas in the image to find potential missing atoms. Each keypoint
        belonging to an open contour looks at its nearest neighbour locations and finds a candidate if the nearest neighbour
        is in an open area (as well as not very close to closed area) and at least two other keypoints from the same open contour 
        see a nearest neighbour at the same location. The average of the position of this group of candidates is taken and classified
        with the previously trained SVM. Accepted candidates are declared to be atoms missed by the SIFT algorithm.
        This algorithm also helps with showing where an atom in a vacancy could be found.
        

        Args:
            acceptance_threshold (float, optional): Threshold parameter indicating point proximity limit
                to be accepted for further clustering. Values between 1. and 2. have proven to be the 
                most efficient, with 1. having the fewest false positives and 2. having the fewest false negatives.
                Defaults to 1..
            border_distance (float, optional): Fraction of the closest distance of two atoms found in a tile which sets a minimum for the 
                distance of an atom to the contour border. Defaults to 2/3.
            decision boundary (float, optional): Sets the minimum distance from the SVC dividing hyperplane to be accepted. Defaults to 0.
        
        Sets:
            accepted_candidates: Candidate keypoints that have been accepted as atoms.
            rejected_candidates: Candidate keypoints that have been rejected (due to e.g. vacancies, misdetection, high acceptance threshold)
            
        Returns:
            None.

        """
        #initialize lists holding the accepted and rejected points
        self.accepted_candidates = []
        self.rejected_candidates = []
        
        self.all_candidates = []
        
        #loop through the 'open' contours
        for t, tile in enumerate(self.kp_open_id):
            
            #get coordinates of the points at tile corners
            tile_points = self.label_kp_coord[tile]
            
            #initialize list of potential candidate points
            candidate_points = []
            
            #loop through the points in the tile
            for tile_point in tile_points:
                
                #for each point in each symmetry direction
                for symmetry_direction in self.vectors:
                    
                    #get the point that the symmetry vector from the tile point points to
                    candidate_point = tile_point + symmetry_direction
                    
                    #check if the point is in an unfilled area 
                    if (int(np.max(candidate_point))<(len(self.filled_tiles_bin)-2)
                        and np.min(candidate_point)>=1
                        and np.max(self.filled_tiles_bin[int(candidate_point[1])-1:int(candidate_point[1])+2,
                                                         int(candidate_point[0])-1:int(candidate_point[0])+2])!=1):
                        
                        #check if the tile point is further away from the contour than the minimum NN distance
                        if np.min(np.linalg.norm(candidate_point - self.open_contours[t],axis=1))>= border_distance*self.min_tile_distance:
                            
                            #if all is passed, point is accepted as a candidate
                            candidate_points.append(candidate_point)
            
            #convert candidate points to numpy array for better indexing
            candidate_points = np.array(candidate_points)
            
            #if no candidate points have been found (or lass than 3, which is not possible), continue loop
            if len(candidate_points) <3:
                continue
            for cp in candidate_points:
                self.all_candidates.append(cp)
            
            #get pairwise distances of candidate points to each other, sorted by distance
            candidate_pairwise = np.sort(np.linalg.norm(candidate_points - candidate_points[:,np.newaxis],
                                                        axis=2),axis=1)[...,1:]
            
            #get minimum (for 10 and 8fold) or second minimum (for 12 fold) distance for each point
            if len(self.vectors)==12:
                pairwise_second_min = np.min(candidate_pairwise[...,1:], axis=1)
            else:
                pairwise_second_min = np.min(candidate_pairwise[...,0:], axis=1)
            
            
            #get indices of points not very close to at least 2 other points
            outlier_indices= np.where(pairwise_second_min>acceptance_threshold * self.median_kp_size)[0]
            
            #remove outlier points
            candidate_points = np.delete(candidate_points, outlier_indices, axis=0)
            self.candidate_points=candidate_points
            
            #set the cluster span depending on points found to avoid errors from too few points
            if len(candidate_points)>5:
                cluster_span=range(2,int(len(candidate_points)/2))
                
                #cluster the points to find groups where symmetry vectors point to same next object
                candidate_labels = find_best_clustering(candidate_points, span=cluster_span)
                
            elif len(candidate_points)>3:
                cluster_span=range(2,3)
                
                #cluster the points to find groups where symmetry vectors point to same next object
                candidate_labels = find_best_clustering(candidate_points, span=cluster_span)
                
            else:
                candidate_labels = np.array([0 for candidate_point in candidate_points])
            
            #initialize list holding the means of the clusters of candidate points
            candidate_means = []
            
            #go through all the different labels
            for label in np.unique(candidate_labels):
                
                #find indices where the chosen label applies
                candidate_cluster_indices = np.where(candidate_labels==label)[0]
                self.candidate_cluster_indices = candidate_cluster_indices
                
                if len(self.candidate_cluster_indices)>=2:
                    
                    #get the mean of the point cluster and append to list
                    candidate_cluster_mean = np.mean(candidate_points[candidate_cluster_indices],axis=0)
                    candidate_means.append(candidate_cluster_mean)
                
            #convert to numpy array
            candidate_means = np.array(candidate_means)   
            
            #list to hold keypoints
            candidate_keypoints= []
            
            #loop through the means
            for candidate_mean in candidate_means:

                #create a keypoint at the candidate point location
                candidate_kp = cv.KeyPoint(candidate_mean[0], candidate_mean[1], size = self.median_kp_size, angle=0)
                candidate_keypoints.append(candidate_kp)
            
            #get descriptor of the new keypoint
            candidate_keypoints, candidate_descriptors = self.sift.compute(self.gray,candidate_keypoints)
            
            for c,_ in enumerate(candidate_means):
                
                #predict with support vector machine
                candidate_pred = self.classifier.predict([candidate_descriptors[c]])
                candidate_decision = self.classifier.decision_function([candidate_descriptors[c]])


                #choose to accept or reject candidates depending on if they were predicted correctly with a certain accuracy
                if candidate_pred and np.linalg.norm(candidate_decision)>decision_boundary:
                    self.accepted_candidates.append([candidate_keypoints[c],candidate_descriptors[c], candidate_decision])  
                
                else:
                    self.rejected_candidates.append([candidate_keypoints[c],candidate_descriptors[c]]) 
        
    def fill_missing_points(self):
        """Missing point filler
        
        Adds the keypoints and descriptors of accepted points (accepted_candidates) to the total list of keypoints/descriptors.

        Returns:
            None.

        """
        
        
        if len(self.accepted_candidates)>1: 
            
            #in case two open tiles share an edge, two keypoints may overlap, so the distances between points are calculated
            kp_locs = np.array([[a_c[0].pt[0],a_c[0].pt[1]] for a_c in self.accepted_candidates])
            kp_scores =  np.array([a_c[2] for a_c in self.accepted_candidates])
            mins = get_pairwise_min_distance(kp_locs)
            
            ac_to_remove=[]
            
            #if any two points are withing close proximity, remove the one with the worse SVC decision score
            for i,min_ in enumerate(mins):
                
                if min_ < self.median_kp_size:
                    mins_idx = np.where(mins==min_)[0]
                    scores = kp_scores[mins_idx]
                    ac_to_remove.append(mins_idx[np.argmin(scores)])
            

            #remove only if "duplicates" were found
            if len(ac_to_remove)!=0:
                ac_indices = np.delete(np.arange(len(kp_scores)), np.unique(ac_to_remove))
            else:
                ac_indices = np.arange(len(kp_scores))
            
            
            self.kp = np.append(self.kp, np.array([a_c[0] for a_c in self.accepted_candidates])[ac_indices])
            self.des = np.vstack((self.des, np.array([a_c[1] for a_c in self.accepted_candidates])[ac_indices]))
            self.kp_labels = np.append(self.kp_labels, np.array([self.kp_label for a_c in self.accepted_candidates])[ac_indices])
        
        elif len(self.accepted_candidates)==1:
            self.kp = np.append(self.kp, np.array([a_c[0] for a_c in self.accepted_candidates]))
            self.des = np.vstack((self.des, np.array([a_c[1] for a_c in self.accepted_candidates])))
            self.kp_labels = np.append(self.kp_labels, np.array([self.kp_label for a_c in self.accepted_candidates]))
            
    ###
    # application functions
    ###

    def compute(self, classifier_kernel='rbf',contour_region = 6., predicted_shape_corners = [3,4,6], 
                max_outlier_deviation = 4., shape_span = range(2,7), max_shape_clusters=3,
                colors = [(255,0,0),(0,255,0),(0,0,255)],
                tile_filter = False, refilled=False):
        """Basic computation 
        
        This method allows for a basic computation, going through the pattern recognition steps with limited
        algorithm tuning choices. While using the methods for each step is encouraged to achieve the best possible
        result, this allows for a quick initial "check" to help with tuning the hyperparameters.
        

        Args:
            classifier_kernel (str, optional): Kernel type for the Support Vector Machine. Defaults to 'rbf'.
            contour_region (Tfloat, optional): Region parameter for the method match_keypoints_to_contours. 
                Defaults to 6..
            predicted_shape_corners (list, optional): List holding the assumed amount of corners the tiles can have.
                Used for match_keypoints_to_contours and get_tile_angles. Defaults to [3,4,6].
            max_outlier_deviation (float, optional): Outlier parameter for shape filtering (filter_outlier_shapes).
                Only used if tile_filter=True. Defaults to 4..
            shape_span (range, optional): Range for the shape clustering. Defaults to range(2,7).
            max_shape_clusters (3, optional): Maximum number of different tiles should be detected. Defaults to 3.
            colors (list, optional): List of RGB values for filling in the tiles in the binary image. Each list entry
                is a tuple of length 3. Defaults to [(255,0,0),(0,255,0),(0,0,255)].
            tile_filter (bool, optional): Choice if the tile filter should be applied to filter out wonky shapes. Defaults to True.
            refilled (bool, optional): Parameter indicating if the point finder algorithm with subsequent
                filling in of the missed atoms was executed prior to calling this method. Defaults to False.

        Returns:
            None.

        """
    	
        if not refilled:
            self.calculate_filtered_keypoints()
            self.calculate_kp_clusters()
        self.get_informative_label_neighbours()
        if not refilled:
            self.get_classifier(kernel=classifier_kernel)
        self.center_locations()
        self.calculate_dominant_distance_vectors()
        self.set_contours()
        self.match_keypoints_to_contours(region=contour_region, 
                                          predicted_shape_corners=predicted_shape_corners)
        self.get_tile_angles(predicted_shape_corners=predicted_shape_corners)
        if tile_filter:
            self.filter_outlier_shapes(max_deviation = max_outlier_deviation)
        self.cluster_shapes(shape_span = shape_span, max_clusters = max_shape_clusters)
        self.get_average_angles()
        self.fill_tiles(colors)
        self.compute_tile_point_locations(show_plot=False)
    
    
    def fill_and_compute(self, acceptance_threshold=1., border_distance=2/3, decision_boundary=0.0, 
                         runs = 1, show_intermittent_plots = False,**kwargs,):
        """Point filling with basic computation
        
        To be executed after computing the patterns for an image first (e.g. using the compute function). Finds any
        missing points and recomputes the patterns.
        
        Args:
            acceptance_threshold (float, optional): Acceptance threshold parameter for 
                find_missing_points(). Defaults to 1.0.
            border_distance (float, optional): Border distance threshold parameter for 
                find_missing_points(). Defaults to 2/3.
            decision_boundary (float, optional): Decision boundary parameter for 
                find_missing_points(). Defaults to 0.0.
            runs (int, optional): How many times . Defaults to 1.
            show_intermittent_plots (bool, optional): Offers a choice of generating plots showing rejected and accepted 
                candidats. Defaults to False.
            **kwargs : Arguments for the compute function

        Returns:
            None.

        """
        
        for run in range(runs):
            
            #find missing points
            self.find_missing_points(acceptance_threshold=acceptance_threshold)
            self.fill_missing_points()
            #stop runs if no new keypoints have been found in the previous round
            if len(self.accepted_candidates) == 0:
                print("No more candidates after",run, "runs.")
                break
            #plot the accepted and rejected candidates
            if show_intermittent_plots:
                fig,ax= plt.subplots()
                self.plot_all_tiles(ax)  
                for a_c in self.accepted_candidates:
                    ax.add_patch(patches.Circle((a_c[0].pt[0],a_c[0].pt[1]), radius = a_c[0].size, 
                                                color='g', fill=False))
                    #ax.scatter(a_c[0].pt[0], a_c[0].pt[1],c='g')
                for r_c in self.rejected_candidates:
                    ax.add_patch(patches.Circle((r_c[0].pt[0],r_c[0].pt[1]), radius = r_c[0].size, 
                                                color='r', fill=False))
                    #ax.scatter(r_c[0].pt[0], r_c[0].pt[1],c='r')

            
            self.compute(refilled=True, **kwargs)
            
        
        
        
            
    ###
    # plotting functions
    ###
    
    def plot_relevant_keypoints(self, ax, cmap='Greys_r',kp_color='lightgreen', which='label'):
        """Keypoint display
        
        Plots all the keypoints or just the keypoints that depict atoms, depending on the choice for which, on top
        of the image.  

        Args:
            ax (matplotlib.axes._axes.Axes): The ax along which to plot. Must be created externally.
            cmap (matplotlib.colors.ListedColormap, optional): Colormap for displaying the grayscale of the original
                image. Type can also be matplotlib.colors.LinearSegmentedColormap or a string naming a 
                matplotlib.colormaps entry. Defaults to 'Greys_r'.
            kp_color (str, optional): Color of the keypoint markers. String has to denote a valid color, see 
                https://matplotlib.org/stable/tutorials/colors/colors.html . Defaults to 'lightgreen'.
            which (str, optional): Must be 'label' or 'all'. Choice if all the keypoints or just the atom-containing
                keypoints should be shown. Defaults to 'label'.

        Returns:
            None.

        """

        
        #plot grayscale image
        ax.imshow(self.gray, cmap=cmap, vmin=0, vmax=255)
        
        #decided which keypoints to plot
        if which=='label':
            idx = np.where(self.kp_labels==self.kp_label)[0]
        else:
            idx = np.arange(len(self.kp_labels))
        
        #for each keypoint, plot a circle with the keypoint size at its location
        for i in idx:
            loc = self.kp[i].pt
            ax.add_patch(patches.Circle((loc[0],loc[1]), radius = self.kp[i].size, 
                                        color=kp_color, fill=False))
    
        
        
    def plot_vectors(self, ax, origin = [0,0], scale=1., scale_units='xy', angles='xy',
                     show_NN = True, NN_cmap='viridis',**kwargs):
        """Vector display
        
        Shows the symmetry vectors along with the clustered nearest neighbour distribution.

        Args:
            ax (matplotlib.axes._axes.Axes): The ax along which to plot. Must be created externally.
            origin (list, optional): List must be of shape (2) and hold the x and y coordinates of the vector origin. 
                Defaults to [0,0].
            scale (float, optional): Scaling of the vectors used for the quiver function. 
                See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html. for more information. 
                Defaults to 1..
            scale_units (str, optional): Argument must be in {'width', 'height', 'dots', 'inches', 'x', 'y', 'xy'}.
                Denotes the type of scaling used in the quiver function. Defaults to 'xy'.
            angles (str, optional): Argument must be in {'uv', 'xy'}. Denotes the method for determining
                the angles of the vectors in the quiver function. Defaults to 'xy'.
            show_NN (bool, optional): Allows to choose between showing the nearest neighbour distribution or not. 
                Defaults to True.
            NN_cmap (matplotlib.colors.ListedColormap, optional): Colormap to be used for coloring the 
                nearest neighbour clusters. Defaults to 'jet'.
            **kwargs : Other parameters for the matplotlib.pyplot.quiver function.

        Returns:
            None.

        """
        
        #plot the nearest neighbour distribution if the parameter is True
        if show_NN:
            
            #get colormap to distinguish between distances
            cmap= plt.get_cmap(NN_cmap,len(np.unique(self.distance_labels)))
            c = np.array(cmap(self.distance_labels))
            
            #plot the points
            ax.scatter(self.locations_NN_centered.reshape((-1,2)).T[0],
                       self.locations_NN_centered.reshape((-1,2)).T[1],
                       alpha=0.1, c=c)
        
        #get vector origins
        origins = np.zeros((len(self.vectors),2))+origin
        
        #plot the vectors as arrows from origin point
        ax.quiver(origins.T[0],origins.T[1],self.vectors.T[0],self.vectors.T[1],
                  scale=scale, scale_units=scale_units, angles=angles, **kwargs)
        
    
    def plot_all_tiles(self,ax,overlay=True,cmap='Greys_r', alpha=0.5, colors = None):
        """Tile display
        
        Shows the quasicrystal pattern with colored tiles, can be plotted over the original image for comparison. 

        Args:
            ax (matplotlib.axes._axes.Axes): The ax along which to plot. Must be created externally.
            overlay (bool, optional): Choice if pattern should be shown as an overlay on the original image
                or not. Defaults to True.
            cmap (matplotlib.colors.ListedColormap, optional): Colormap for displaying the grayscale of the original
                image. Type can also be matplotlib.colors.LinearSegmentedColormap or a string naming a 
                matplotlib.colormaps entry. Defaults to 'Greys_r'.
            alpha (float, optional): Opacity of the tiles. Value must be between 0 and 1. Defaults to 0.5.
            colors (list, optional): Colors of the tiles given in (R,G,B) value tuples. None uses the colors 
                from rgb_tile_lines.
                If argument is passed, it overrides previously set colors of rgb_tile_lines.
                Defaults to None.

        Returns:
            None.

        """
        if colors is not None:
            #convert the binary contour image to RGB
            rgb_tile_lines = cv.cvtColor(self.connections,cv.COLOR_GRAY2RGB)
            
            #convert again to image object
            rgb_tile_lines = PIL.Image.fromarray(rgb_tile_lines)
            
            #get the midpoint of each valid tile
            tile_midpoints = np.array([np.mean(self.label_kp_coord[kp_tile],axis=0) 
                                        for kp_tile in self.kp_tile_id]).astype(np.int32)
            
            #draw unto image:
            for t, tile in enumerate(tile_midpoints):
                
                PIL.ImageDraw.floodfill(rgb_tile_lines, tuple(tile), colors[self.shape_labels[t]])
            
            #convert back to array
            self.rgb_tile_lines = np.asarray(rgb_tile_lines)
        
        #show image if chosen to
        if overlay:
            ax.imshow(self.gray, cmap=cmap, vmin=0, vmax=255)
            
        ax.imshow(self.rgb_tile_lines, alpha=alpha)
        
    def compute_tile_point_locations(self, show_plot=False, tile_colors = ['r','g','b'], **kwargs):
        """Tile corner position calculator
        
        Computes the positions of the tile corners in order to display the individual average tiles as building blocks
        of the pattern as a whole. Integrated plotting if show_plot is True.

        Args:
            show_plot (bool, optional): Option to create a plot upon calculation or not. Defaults to True.
            tile_colors (list, optional): List of strings giving matplotlib colors for plotting. Each shape cluster is assigned one 
                color. Defaults to ['r','g','b'].
            **kwargs : Further arguments for the matplotlib.pyplot.subplots function.

        Returns:
            None.

        """
        
        
        
        #get unique shape labels
        unique_shape_labels, unique_counts = np.unique(self.shape_labels, return_counts=True)
        
        #initialize list of tile corner points locations
        self.tiles_in_points = []
        
        #list for annotation angles used later in the plots
        annotation_angles = []     
        
        #loop through each tile to find the connecting vectors (direction+length)
        for a, angles in enumerate(self.average_angles):
            
            #get rid of any zeroes
            angles = np.delete(angles, angles==0)
            
            #take averages of opposite angles for shapes with an even number of corners
            if not len(angles)%2:
                
                #reshape to get opposite angles of same 
                angles = angles.reshape(-1,2)
                
                #set opposites same to mean general angle 
                angles = np.mean(angles, axis=1)
                
                #stack twice together to get the angles in a perf_counterwise fashion
                angles = np.hstack((angles, angles))
            
            #take total average for odd number of corners and set each angle to that
            else:
                
                #get mean and stack
                angle_mean = np.mean(angles)
                angles = [angle_mean for a in angles]
            
            annotation_angles.append(angles)
            
            #set list holding directional vectors and the tile points (with origin as first point)
            angle_vectors = []
            tile_points = [np.array([0,0])]
            
            #get directional vector for each angle (except the last one where the path is clear)
            for i, angle in enumerate(angles[:-1]):
                
                #for the first item take 90-half the angle to rotate 
                if not i:
                    
                    #set the first tile point to be origin
                    tile_origin = np.array([0,0])
                    
                    #set the unity vector with average distance between points to be along x axis
                    unity_vector = np.array([self.vector_length,0])
                    
                    #get rotation angle
                    rotation_angle = np.deg2rad(90-angle/2)
                
                else:
                    
                    tile_origin = tile_points[i]
                    
                    #set unity vector to be previous directional vector
                    unity_vector = angle_vectors[i-1]
                    
                    #rotation angle is the current angle subtracted from 180 degrees 
                    rotation_angle = np.deg2rad(180-angle)
                    
                #apply rotation
                rotation_matrix = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle)],
                                            [np.sin(rotation_angle),np.cos(rotation_angle)]])
                
                #apply rotation matrix to get vector
                vec = rotation_matrix@unity_vector
                
                #add vector to origin to receive new tile point
                tile_point = tile_origin+vec
                
                #append results to list
                angle_vectors.append(vec)
                tile_points.append(tile_point)
            
            #include end point (same as start point)
            tile_points.append(np.array([0,0]))

            #append to final list
            self.tiles_in_points.append(np.array(tile_points))
        
        #get the minimum distance between points in the tiles
        self.min_tile_distance = np.min([np.sort(np.unique(np.linalg.norm(tile-tile[:,np.newaxis],axis=2)))[1] for tile in self.tiles_in_points])

        if show_plot: 
            #plot each tile
            fig,ax= plt.subplots(ncols=len(unique_shape_labels), sharey=True,**kwargs)
            fig.tight_layout()
            
            for t, tile in enumerate(self.tiles_in_points):
                plot_tile = np.array(tile).T
                plot_color = tile_colors[unique_shape_labels[t]]
                ax[t].set_title("Count:"+str(unique_counts[t]))
                ax[t].set_aspect('equal')
                ax[t].fill(plot_tile[0], plot_tile[1], c=plot_color)
                ax[t].axis('off')
                
                #add angle display
                for a,angle in enumerate(annotation_angles[t][:-(len(annotation_angles[t])%2)+2]):
                    ax[t].annotate(str(np.round(angle,2))+"°",
                                   (plot_tile[0][a], plot_tile[1][a]), 
                                   path_effects=[patheffects.withStroke(linewidth=2, foreground="w")])
        
    
    ###
    # getter functions
    ###
    
    def get_keypoint_coordinates(self):
        """Keypoint coordinate getter
        
        Gets the coordinates of the keypoints that are associated with atoms. Can be used for other 
        scientific purposes.

        Returns:
            array: Keypoint coordinate given as an array of shape (n,2)

        """
        return self.label_kp_coord
    
    def get_tile_image(self):
        """Pattern image getter
        
        Gets the binary image with filled in tiles.         
        
        Returns:
            array: RGB image containing the connection between atoms and the color-coded tiles.

        """
        return self.rgb_tile_lines


    

