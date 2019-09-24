#Young Object Discoverer Algorithm (YODA)

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import regularizers
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
from keras.utils import plot_model
import matplotlib.backends.backend_pdf 
from keras import regularizers
from astropy.io import fits
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn import preprocessing
import time
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn import preprocessing
from sklearn.utils import resample
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from google.colab import files
from google.colab import drive
!pip install keras-metrics
import keras_metrics as km
import csv
import os
import tempfile
from keras.callbacks import Callback
drive.mount('/content/gdrive')

# Install the PyDrive wrapper & import libraries.
# This only needs to be done once in a notebook.
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive2 = GoogleDrive(gauth)

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive

#gauth = GoogleAuth()
# Try to load saved client credentials
#gauth.LoadCredentialsFile("mycreds.txt")
#if gauth.credentials is None:
    # Authenticate if they're not there
#    gauth.LocalWebserverAuth()
#elif gauth.access_token_expired:
    # Refresh them if expired
#    gauth.Refresh()
#else:
    # Initialize the saved creds
#    gauth.Authorize()
# Save the current credentials to a file
#gauth.SaveCredentialsFile("mycreds.txt")
