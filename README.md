Young Objects Discovery Algorithm (YODA)

Neural Network based algorithm to identify new high-mass Pre-Main Sequence objects (Herbig Ae/Be stars) within Gaia, AllWISE, IPHAS and VPHAS+ (Input Sample, see Vioque et al. in prep). The algorithm can be adapted to other target sources and catalogues. Most of the hyper-parameters and input lists can be edited at the very beginning of the code.

In short, the algorithm is envisaged to assess the problem of disentangling two very similar classes (Herbig Ae/Be, category 1, and Classical Be stars, category 2). This is a very skewed problem as the number of these sources is negligible with respect to the size of the Input Sample. In addition, we know very few of them for the training. The issues of having a very a small training set and aiming at categorizing very skewed classes are addressed by means of bootstrapping and weighting the classes in a balanced fashion. This can be edited in order to aim to other classification problems.

The pipeline of the algorithm can be structured as follows:

1st - Selection of known sources for training and selection of Input Sample of unknown sources to classify (from the latter a small random fraction will be withdrawn to form a third category of "other" sources for the Training)

2nd - Selection of the characteristics

4rd - Generation of bootstrapped samples

3rd - Principal Component Analysis (PCA) to the characteristics to obtain a set of features (keep 99.99% of variance)

5th - Weighting of the skewed classes

6th - Neural Network classification. Training stops when Cross-Validation precision gets to a maximum.

7th - Evaluation on Test Set

8th - Classification and generation of output catalogues

The output of YODA is a file per bootstrapped iteration with the classification, i.e., the Input Sample with three more columns, each one with the probability of the object of belonging to each one of the input categories: Herbig Ae/Be, Classical Be or something else. In order to average the bootstrapped sets and get a final catalogue, the Bootgather.py code can be used.
