Young Objects Discovery Algorithm (YODA)

Neural Network based algortihm to identify new high-mass Pre-Main Sequence objects (Herbig Ae/Be stars) within Gaia, AllWISE, IPHAS and VPHAS+ (Input Sample, see Vioque et al. in prep). The algorithm can be adapted to other target sources and catalogues. Most of the hyper-parameters and input lists can be edited at the very beginning of the code.

In short, the algorithm is envisaged to assess the problem of disentangling two very similar classes (Herbig Ae/Be and Classical Be stars). This is a very skewed problem as the number of these sources is negligible with respect to the size of the Input Sample. In addition, we know very few of them for the training. The issues of having very a small training set and aiming at categorizing very skewed classes are addresed by means of bootstrapping and weighting the classes in a balanced fashion. This can be edited in order to aim to other classification problems.

The pipeline of the algorithm can be structured as follows:

1st - Selection of known sources for training and Input Sample of unkwown sources to classify (from the latter a small fraction will be taken to form a category of other random sources for the Training).

2nd - Selection of the characteristics

4rd - Generation of Bootstrapped samples

3rd - Principal Component Analysis (PCA) to the characteristics to obtain a set of features

5th - Weighting of the skewed classes

6th - Neural Network classification

7th - Evaluation on Test Set

8th - Classification and generation of output catalogues

The output of YODA is a file per bootstrapped iteration with the classifications for the probability. In order to average the bootstrapped sets and get a final catalogue the Bootgather.py code can be used.
