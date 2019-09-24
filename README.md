Young Objects Discovery Algorithm (YODA)

Neural Network based algortihm to identify new high-mass Pre-Main Sequence objects (Herbig Ae/Be stars) within Gaia, AllWISE, IPHAS and VPHAS+ (see Vioque et al. in prep). The algorithm can be adapted to other target sources and catalogues.

The pipeline of the algorithm can be structured as follows:

1st - Selection of known sources for training and Input Sample of unkwown sources to classify

2nd - Selection of the characteristics

4rd - Generation of Bootstrapped samples

3rd - Principal Component Analysis (PCA) to the characteristics to obtain a set of features

5th - Weighting of the skewed classes

6th - Neural Network classification

7th - Evaluation on Test Set

8th - Classification and generation of output catalogues

In short, the algorithm is envisaged to tackle the problem of disentangling two very skewed classes (Herbig Ae/Be and Classical Be stars) with respect to the whole Input Sample. 
