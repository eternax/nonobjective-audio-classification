# nonobjective-audio-classification
Deep learning model for the non-objective classification of audio signals

by Guillemgp:

Installs: 

To run the code the following libraries must be installed using pip install:

sklearn
tensorflow.keras
librosa
pyAudioAnalysis
soundfile
matplotlib

Preparation of the data: 

Sound effects used for training are to be placed inside ’00_raw_sfx’ folder found inside the ‘raw_data’ folder:

'.../nonobjective-audio-classification/raw_data/00_raw_sfx'

Sound textures used for training are to be placed inside 00_raw_texture folder found inside the raw_data folder:

'.../nonobjective-audio-classification/raw_data/01_raw_textures’

Prepared data is saved inside the folder ‘data’:

'.../nonobjective-audio-classification/data/00-Sfx’

'.../nonobjective-audio-classification/data/01-Textures’

_____________________________________________________________________________________________

The first time the scripts are run:

The path for these four directories must be added to the data_prep.py script and the pred_data_prep.py script 
lines 22, 24, 35 and 37.
_____________________________________________________________________________________________

The scripts should be run in the terminal in the following logical order: 

python data_prep.py                  # prepares the training audio files.

python predict_data_prep.py     # prepares the prediction audio files

python feat_extract.py               # extracts features of the training data    

python feat_extract_pred.py      # extracts features of the prediction data

python cnn.py -t                        # train files

python cnn.py -p                       # predict files

python evaluation_results.py     # evaluates the results


Complementary files folder: 
A trained model is included under the name ’trained_model.h5’
Diverse materials such as .npy and .csv files are given to give a deeper insight into the feature results and data. 

