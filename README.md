Young Object Discoverer Algorithm (YODA)

Artificial Neural Network-based algorithm designed to identify new high-mass Pre-Main Sequence (PMS) objects (Herbig Ae/Be stars) within Gaia, AllWISE, IPHAS and VPHAS+ (Sample of Study, see Vioque et al. 2020). The algorithm can be adapted to other sources and catalogues. Most of the hyper-parameters and input sets can be edited at the very beginning of the code.

In short, the algorithm is designed to disentangle two very similar classes of objects (PMS-Herbig Ae/Be, category 1, and classical Be stars, category 2) and in general to find new high-mass PMS objects. This is a very skewed problem as the number of sources of these classes is negligible with respect to the size of the Sample of Study. In addition, very few of them are known that can be used for the training. These issues of having a very small training set and aiming at categorizing very skewed classes are addressed by means of bootstrapping and weighting the classes in a balanced fashion. This can be edited in order to aim for other classification problems.

The pipeline of the algorithm is structured as follows:

1st - Selection of known sources for the training and selection of an Input Set of unknown sources to classify (from the latter a small random fraction is withdrawn to form a third category for the training of "other" sources, category 0)

2nd - Selection of the characteristics

3rd - Generation of bootstrapped samples

4th - Principal component analysis (PCA) is applied to the characteristics to obtain a set of features (keeps 99.99% of variance).

5th - Weighting of the skewed classes

6th - Artificial Neural Network classification. Training stops when the precision of the category 1 on cross-validation set gets to a maximum.

7th - Evaluation on test set

8th - Classification of Input Set sources with the trained network

The output of YODA is a file per bootstrapped iteration with the corresponding artificial neural network classification. This is, the Input Set with three more columns, each one with the probability of the object of belonging to each one of the input categories: other, PMS or classical Be. In addition, YODA outputs the metrics of precision and recall for category 1 and 2 from evaluation on test set. In order to average the resulting probabilities of the bootstrapped sets and obtain a final catalogue, the Bootgather.py code can be used. For further details see Vioque et al. 2020.
