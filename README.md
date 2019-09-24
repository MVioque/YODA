Young Objects Discovery Algorithm (YODA)

Neural Network based algortihm to identify new high-mass Pre-Main Sequence objects (Herbig Ae/Be stars) within Gaia, AllWISE, IPHAS and VPHAS+ (Input Sample, see Vioque et al. in prep). The algorithm can be adapted to other target sources and catalogues.

In short, the algorithm is envisaged to tackle the problem of disentangling two very skewed classes (Herbig Ae/Be and Classical Be stars) with respect to the whole Input Sample. The issues of having very a small training set at aiming at categorizing very skewed classes was addresed by means of bootstrapping and weighting the classes in a balanced fashion. This can be edited in order to aim for different problems.

The pipeline of the algorithm can be structured as follows:

1st - Selection of known sources for training and Input Sample of unkwown sources to classify (from the latter a small fraction will be taken to form a category of other random sources for the Training).

2nd - Selection of the characteristics

4rd - Generation of Bootstrapped samples

3rd - Principal Component Analysis (PCA) to the characteristics to obtain a set of features

5th - Weighting of the skewed classes

6th - Neural Network classification

7th - Evaluation on Test Set

8th - Classification and generation of output catalogues

The output of YODA will be a probability assignment to each ot the sources in the Input Sample for each of the bootstrapped sets. In order to average the bootstrapped sets and get a final catalogue the Bootsum.py code can be used.
