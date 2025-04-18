# AiSurf: *A*utomated *I*dentification of *Surf*ace images
AiSurf is an open-source package for analizing surface microscopy images. New functionalities are added with time.
The main advantage of AiSurf is that it doesn't require any image database for training, which is a bottleneck for many image classification programs. No programming skills are required to use this tool, only the istructions written in the *Usage* sections of the respective notebooks need to be followed. <br>

**These are the current available methods:** <br>
1. *Lattice Extraction* aims to inspect and classify atomically-resolved images (like AFM and STM) via Scale Invariant Feature Transform [(SIFT)](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) and Clustering Algorithms, inspired by the work of [Laanait et al](https://ascimaging.springeropen.com/articles/10.1186/s40679-016-0028-8). <br>
Lattice Extraction extracts primitive lattice vectors, unit cells, and structural distortions from the original image, with no pre-assumption on the lattice and minimal user intervention.

2. *Atom Counting* allows to count features on images through a template-matching procedure. It is a natural extension of Lattice Extraction and it is typically used to count atoms on surface images.

3. *QuasiCrystal Pattern Extractor (QCP)* extracts the tiles from images displaying quasicristalline patterns. The tiling along with a statistical analysis can be obtained.

4. *Total Variation decomposition* allows to smooth, decompose or denoise (microscopy) images using total variation minimization techniques.


## Related works
We kindly ask the user to cite the articles relative to AiSurf's functionalities when using them for their scientific research.

[Lattice Extraction related article](https://iopscience.iop.org/article/10.1088/2632-2153/acb5e0/meta).<br>
*Marco Corrias et al 2023 Mach. Learn.: Sci. Technol. 4 015015,* **DOI:** *10.1088/2632-2153/acb5e0* <br>

[Atom Counting related article](https://pubs.acs.org/doi/10.1021/acsami.4c13795).<br>

[Quasicristalline pattern recognition related article](https://arxiv.org/abs/2503.05472). (under revision) <br>

[Total Variation decomposition](xxx). (under submission, link soon available.) <br>


## Installation
No installation is needed, the user just needs to download this repository.


## <a name="usage_lattice_extr"></a> Usage - Lattice Extraction

### Dependencies
* NumPy
* Matplotlib
* SciPy
* Scikit-learn (sklearn)
* Python Image Library (PIL)
* OpenCV

### General setup
In order to start the lattice recognition process, image and simulation parameters need to be set. This can be done in the following ways:
* Create a folder where image, parameters file and results will be stored. In this repository, such folders are inside the [examples](https://github.com/QuantumMaterialsModelling/Lattice-Symmetry-Recognition/tree/master/examples) folder;
* Specify the path (relative to the notebook) and the image name at the beginning of the IPython notebook [lattice_extraction.ipynb](https://github.com/QuantumMaterialsModelling/Lattice-Symmetry-Recognition/blob/master/lattice_extraction.ipynb). For example, the third cell of the notebook reads: <br>
```
# Insert path + filename here:
path = "examples/lattice_extraction/SrTiO3(001)/"
filename = "small SrTiO3_1244.png"
```

### <a name="parameters_lattice_extr"></a> Parameters file setup
The parameters file, *parameters.ini* is the file containing all the parameters needed to run the simulation. It must be put inside the image folder, but if not provided some default parameters will be used instead; such parameters are found at the beginning of the IPython Notebook file. This section will describe the meaning of each parameter; suggestions regarding the parameter tuning are inserted in the Notebook, just before they are used. Images in the [examples](https://github.com/QuantumMaterialsModelling/Lattice-Symmetry-Recognition/tree/master/examples) folder of this repository can also be used as a reference for parameter tuning.

[*SIFT*] <br>
Three main parameters of the SIFT algorithm, well explained in the [original article](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) by Lowe and in [this link](https://docs.opencv.org/4.5.4/d7/d60/classcv_1_1SIFT.html).
- **contrast_threshold**: the contrast threshold used to filter out weak features. Higher threshold means more discarded features. *Default: 0.003*;
- **sigma**: the sigma of the Gaussian applied to the input image at the first octave. *Default: 4*;
- **noctavelayers**: the number of layers in each octave. The number of octaves is computed automatically from the image resolution. *Default: 8*.

[*Keypoint filtering*] <br>
These are thresholds to filter out [keypoints](https://paperswithcode.com/task/keypoint-detection) ("kp") that could cause issues in the lattice identification process, in units of the median keypoint size.
- **size_threshold**: if kp_size > median*size_threshold or kp_size < median/size_threshold the keypoint is deleted. *Default: 2*;
- **edge_threshold**: all keypoints that are closer than median\*edge_threshold to one border of the image are deleted. *Default: 1*.

[*Keypoint Clustering*] <br>
Clusterings with *n* clusters between lower and upper bound are evaluated with respect to their [silhouette score](https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c); the one with the maximal silhouette score is chosen for further processing.
- **cluster_kp_low** and **cluster_kp_high**: values defining the interval containing the optimal number of different clusters found in the image, evaluated by calculating the silhouette score. *Default: 2 and 12 respectively*, they define the variable *clustering_span_kp*;
- **cluster_choice**: number that selects the chosen reference cluster for the second part of the analysis. The value of *1* indicates the *first*/*most populated cluster*, so *2* selects the *second* most populated one and so on. *Default: 1*.

[*Nearest Neighbours*] <br>
Parameters related to the clustering processes used to find the primitive vectors.
- **cluster_kNN_low** and **cluster_kNN_high**: values defining the interval containing the optimal number of clusters for the calculated nearest neighbours (NN) distances, evaluated by the silhouette score. *Default: 6 and 24 respectively*; they define the variable *cluster_span_kNN*. <br>
**cluster_kNN_low** is also the number of NN considered for each keypoint during the calculations.
- **clustersize_Threshold**: used to reduce impact of erroneous NN-vectors on the selection of the lattice vectors. In the final distribution only nn-clusters with population ≥ clustersize_threshold\*n_max are considered; n_max is the population of the largest cluster; *Default: 0.3*.

[*Sublattice lookup*] <br>
Once the primitive vectors have been found, we look for the sublattice positions.
- **cluster_SUBL_low** and **cluster_SUBL_high**: values defining the interval containing the optimal number of sublattice positions. *Default: 2 and 6 respectively*; they define the variable *clustering_span_SUBL*.

[*Deviation plot*] <br>
Parameters related to the perfect-lattice-deviations plot.
- **k2**: number of nearest neighbors considered for each keypoint. *Default: 10*;
- **rtol_rel**: all vectors that are within the relative_r-tolerance of the lattice vectors are drawn; *Default: 4* (pixels);
- **arrow_width**: the arrow_width can be specified (see [matplotlib.quiver() - width parameter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html)). *Default: 0.003*;
- **c_max_arrow**: deviation (in pixels) of a lattice vector with respect to the predicted one. Needed to tune the visualization of bond deviations, purely aesthetic. *Default: None*.


### Example - Lattice Extraction
SrTiO3 (001) with Sr vacancies, calculated with the default parameters written above.
See the [Examples folder](https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/tree/master/examples/lattice_extraction/) for more informations (e.g. parameters setting) and examples. <br>
Keypoints localization after cleaning: <br>
<img src="https://github.com/QuantumMaterialsModelling/Lattice-Symmetry-Recognition/blob/master/examples/lattice_extraction/SrTiO3(001)/example_cleankp.png" width="300" height="300">
<br> Nearest neighbours distances folded into the unit cell: <br>
<img src="https://github.com/QuantumMaterialsModelling/Lattice-Symmetry-Recognition/blob/master/examples/lattice_extraction/SrTiO3(001)/sublattice_positions.png" width="300" height="300">
<br> Arrows connecting Sr atoms, with colours based on their deviation from the primitive vector: <br>
<img src="https://github.com/QuantumMaterialsModelling/Lattice-Symmetry-Recognition/blob/master/examples/lattice_extraction/SrTiO3(001)/deviations_1.png" width="300" height="300">
<br> Final prediction of the cell symmetry: <br>
<img src="https://github.com/QuantumMaterialsModelling/Lattice-Symmetry-Recognition/blob/master/examples/lattice_extraction/SrTiO3(001)/symmetry_cell_average.png"  width="200" height="200">


---


## <a name="usage_atom_counting"></a>Usage - Atom Counting
Refer to the [Usage](#usage_lattice_extr) section of Lattice Extraction.

### Parameters file setup
Other than the [parameters](#parameters_lattice_extr) introduced for Lattice Extraction, new ones have been introduced.

[*Atom count*] <br>
- **r_rescale**: rescales the median features' radius. Used to variate the impact of the atomic neighborhood on the correlation map. Recommended value: between 0.8 and 1.2.
- **min_correl**: filter out the peaks below this value from the correlation map. Use to reduce the outliers from the peak detection/atom counting. Recommended value: between 0.4 and 0.6.
- **d_rescale**: recales the accepted minimum distance between the features, which is equal to half of the median crop's size. The features' filtering process takes care of most of the outliers, and the default value of 0.8 never needed to be modified.

### Example - Atom Counting
See the [Examples folder](https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/tree/master/examples/atom_counting) for more informations (e.g. parameters setting) and examples. <br>
<img src="https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/blob/master/examples/atom_counting/small_SrTiO3_1244/count147.png" width="300" height="300">


---


## <a name="usage_qcp"></a> Usage - QuasiCrystal Pattern Extractor (QCP)

This module was made for a generalized pattern recognition in atomic resolution images of quasicrystal surfaces. 
It is simple to use, needing only a few lines to recognize quasiperiodic patterns (if any are present) as found in quasicrystals, while also
offering a wide range of customizability (as explained in the "Application" section).
Along the tiling, it also detects atom locations, rotational symmetry and the individual tile quantities. 


### Dependencies
 - Numpy
 - Matplotlib
 - OpenCV
 - Pillow (PIL)
 - SciPy
 - Scikit-learn


### Path setup
The easiest way to open an image is to declare the path to the folder and state the filename of the image.  
```
path = ".../imagefolder/"
filename = "image.png"
```

### Parameter setup

If certain class instance parameters are to be changed, it is good to have a *parameters.ini* file in the same folder of the image. It will automatically be considered when the path is given.
The following parameters may be changed in the *parameters.ini* file:

[*SIFT*] <br>
- **c_thr:** The contrast threshold value for the image features. Lowering this means that more image features with a low contrast change relative to the surrounding are discarded.;
- **e_thr:** A threshold for discarding features along edges. Higher threshold means less edge detections.;
- **sigma:** Sigma of the initial Gaussian that is convolved with the image.;
- **n_oct_layers:** The number of layers per octave. Increasing this leads to more keypoint detections, useful for multiple smaller features. 

[*Keypoint filtering*] <br>
- **size_threshold:** Keypoints with sizes >(median size*size_threshold) are discarded. Keypoints with sizes <(median_size/size_threshold) are discarded as well.;
- **edge_threshold:** Keypoints too close to the edge are discarded. The distance to the center of the keypoint from the edge must be <(median size*edge_threshold).;
- **shrink_factor:** Remaining keypoint sizes are set to shrink_factor*median size. This shifts the importance in the descriptors to the central gradient.

[*Nearest Neighbours*] <br>
- **kNN_amount:** How many neighbours should be consided during the nearest neighbour search.

For more info on SIFT parameters, see [here](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)

### Instance initialization

Upon loading the module and setting the path and filename, a class instance can be created using

```
qcp = QCP.QCP(path=path, filename=filename)
```

### Performing the recognition

A practical example can be followed in the Python notebook. <br>
The initial tiling computation can be performed with

```
qcp.compute()
```

Upon completion, the tiling can be displayed by passing

```
fig, ax = plt.subplots()
qcp.plot_all_tiles(ax)
```

### Filling in missing points

As SIFT struggles with some detections, it is good to employ the point search algorithm by using

```
qcp.fill_and_compute(runs=1)
```

where runs can be set arbitrarily high, as the algorithm stops when no more missing points can be found. The final tiling can then be displayed using the method shown above again. 


### Example - Quasicrystalline pattern extractor
See the [Examples folder](https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/tree/master/examples/quasicrystalline_pattern_recognition/Ba-Ti-O_Pt(111)) for more informations (e.g. parameters setting). <br>
<p float="left">
<img src="https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/blob/master/examples/quasicrystalline_pattern_recognition/Ba-Ti-O_Pt(111)/Ba-Ti-O_Pt(111)_tiles.png" width="300" height="300"/>
<img src="https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/blob/master/examples/quasicrystalline_pattern_recognition/Ba-Ti-O_Pt(111)/Ba-Ti-O_Pt(111)_NN.png" width="300" height="300"/>
<img src="https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/blob/master/examples/quasicrystalline_pattern_recognition/Ba-Ti-O_Pt(111)/Ba-Ti-O_Pt(111)_tilecount.png" width="300" height="220"/>
</p>


---


## <a name="usage_qcp"></a> Usage - TV decomposition
### General setup
In order to start processing the image(s), the simulation parameters need to be set. This can be done in the following way:
* create a *param_dn.ini* file and store it in the same folder of *denoising.ipynb*. *param_dn.ini* contains the relative path and the name of the image(s) to be processed, and all the other parameters involved in the workflow. <br>

### Parameters file setup
The parameters file, *param_dn_.ini* is the file containing all the parameters needed to run the simulation. If some parameters are not provided some default ones will be used instead; such parameters are found at the beginning of the IPython Notebook file. This section will describe the meaning of each parameter; examples of parameters settings can be found in the code's repository.

[*files*] <br>
- **path**: folder path relative to the notebook where the raw image is stored;
- **filename**: filename: name of the image(s) to be processed;
- **extension**; file extension.

[*parameters*] <br>
- **iterations**: maximun number of iterations before the calculations stop (in case of convergence problems). *Default*: 2000.
- **err**: root mean square error between the input and output image, below which the calculations stop and the convergence is reached. *Default*: 5E-6.
- **nabla_comp**: components of the gradient operator to be considered. *both*: isotropic calculations; used when scratches or other artifacts imaged as parallel lines are not present od are not wanted to be removed. *x*: considers only horizontal components. Needed when horizontal lines need to be removed, as the vertical signal jumps they lead to are removed. *y*: considers only the vertical gradient components. Used when unwanted, vertical lines are present. *Default*: *both*.
- **algo**: which method to use. 1 - TV-L1, 2 - Huber-ROF, 3 TGV-L1. *Default*: *2 (Huber-ROF)*.
- **lam**: value of the lambda parameter. *Default*: 0.01.
- **alpha**: value of the alpha parameter in the Huber-ROF method. *Default*: 0.05.
- **alpha1 and alpha2**: values of the alpha1 and alpha2 parameters in the TGV-L1 method. *Default*: 1, 2 respectively.
- **gif_yn**: True or False, whether one wants to generate a frame for a gif every 100 iterations. *Default*: False.



### Example - Total Variation Decomposition

See the [Examples folder](https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/tree/master/examples/total_variation_decomposition) for more informations (e.g. parameters setting) ad examples. <br>

<img src="https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/blob/master/examples/total_variation_decomposition/LSMO110.png" width="1052" height="700"/>
<p float="left">
<img src="https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/blob/master/examples/total_variation_decomposition/Mica/Mica_02007_countBlack.png" width="350" height="350"/>
<img src="https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images/blob/master/examples/total_variation_decomposition/Mica/Mica_02007_countBlack_hROF_nabla-both_it500_lam0.5_alpha0.05_median.png" width="350" height="350"/>
</p>