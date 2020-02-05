Young Objects Discovery Algorithm (YODA)

Neural Network-based algorithm to identify new high-mass Pre-Main Sequence (PMS) objects (Herbig Ae/Be stars) within Gaia, AllWISE, IPHAS and VPHAS+ (Sample of Study, see Vioque et al. in prep). The algorithm can be adapted to other target sources and catalogues. Most of the hyper-parameters and input lists can be edited at the very beginning of the code.

In short, the algorithm is envisaged to assess the problem of disentangling two very similar classes (PMS/Herbig Ae/Be, category 1, and Classical Be stars, category 2). This is a very skewed problem as the number of these sources is negligible with respect to the size of the Sample of Study. In addition, very few of them are known for the training. The issues of having a very small training set and aiming at categorizing very skewed classes are addressed by means of bootstrapping and weighting the classes in a balanced fashion. This can be edited in order to aim for other classification problems.

The pipeline of the algorithm can be structured as follows:

1st - Selection of known sources for training and selection of Input Set of unknown sources to classify (from the latter a small random fraction is withdrawn to form a third category of "other" sources, category 0, for the training)

2nd - Selection of the characteristics

4rd - Generation of bootstrapped samples

3rd - Principal Component Analysis (PCA) to the characteristics to obtain a set of features (keep 99.99% of variance)

5th - Weighting of the skewed classes

6th - Neural Network classification. Training stops when cross-validation precision on category 1 gets to a maximum

7th - Evaluation on test set

8th - Classification of Input Set sources with the trained network

The output of YODA is a file per bootstrapped iteration with the resulting neural network classification, i.e., the Input Set with three more columns, each one with the probability of the object of belonging to each one of the input categories: other, PMS or Classical Be. In addition, YODA outputs the precision and recall metrics from evaluation on test set for category 1 and 2. In order to average the probabilities of the bootstrapped sets and get a final catalogue the Bootgather.py code can be used.
