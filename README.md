Young Objects Discovery Algorithm (YODA)

Neural Network based algortihm to identify new high-mass PMS objects (Herbig Ae/Be) within Gaia, AllWISE, IPHAS and VPHAS+ (see Vioque et al. in prep). The algorithm can be adapted to other target sources and catalogues.

The pipeline of the algorithm can be structured as follows:

1st - Selection of Known Sources for Training and Input Sample of unkwown sources to classify
2nd - Selection of the characteristics
3rd - Principal Component Analysis (PCA) to the characteristics to obtain set of features
4rd - Generation of Bootstraped samples
5th - Neural Network
6th - Evaluation on Test Set
7th - Generation of output catalogues
