#!/usr/bin/env python
# coding= UTF-8
# Non-objective-audio-classification Guillermo G. P.
# 20 January 2020
#
# This code has been inspired by the work “Learning Sound Event Classifiers from
# Web Audio with Noisy Labels”, arXiv preprint arXiv:1901.01189, 2019
#

#Import modules and packages:

import argparse
import code
import numpy as np
from glob import glob
import librosa as librosa
import os.path

##### Prepare Prediction data: #####

# Directory for raw sfx folder
data_dir_psfx = '/Volumes/LaCie/project_Guillermo/Sent_project/Project_Guillermo_Garcia_Peeters_Student_170757386_3/raw_predict_data/00_raw_sfx' # add here the path of the pre-processed data.
# Directory for raw textures folder
data_dir_ptextures = '/Volumes/LaCie/project_Guillermo/Sent_project/Project_Guillermo_Garcia_Peeters_Student_170757386_3/raw_predict_data/01_raw_textures' # add here the path of the pre-processed data.


predict_sfx_files = glob(data_dir_psfx + '/*.wav') # find files of interest.
predict_text_files = glob(data_dir_ptextures + '/*.wav') # find files of interest.

print ('nº of predict sfx files =' , len(predict_sfx_files))
print ('nº of predict texture files =' , len(predict_text_files))


# Output directory for prepared predict sfx data:
out_data_dir_psfx = '/Volumes/LaCie/project_Guillermo/Sent_project/Project_Guillermo_Garcia_Peeters_Student_170757386_3/predict/00-Sfx' # add here the output data path
# Output directory for prepared predict texture data:
out_data_dir_ptextures = '/Volumes/LaCie/project_Guillermo/Sent_project/Project_Guillermo_Garcia_Peeters_Student_170757386_3/predict/01-Textures' # add here the output data path


def pData_prep():

    fs = 44100 # sample rate
    print ('fs :', fs)
    for file in range(0, len(predict_sfx_files), 1):
    
        # Load SFX, downsample to 44.100Hz, with an offset of 1sec., load first 60 sec.
        audio_sfxpredict, sfreq = librosa.load(predict_sfx_files[file] , sr = fs , mono = 0, offset = 1.0, duration=60.0)
        basefilename = os.path.basename(predict_sfx_files[file])  # store the filenames
        audio_in = librosa.util.fix_length(audio_sfxpredict, fs*60,axis=-1) # trim or zeropad to 60 sec.
        
        # Get the nº of channels:
        if audio_in.ndim > 1: num_chan = audio_in.shape[0]
        else: num_chan = audio_in.ndim
        
        print ('preparing file: ', basefilename)
        print ('num_chan: ', num_chan)
        
        # If a multichannel file is found (more than 2 channels) it is converted to stereo:
        if num_chan > 2:
            print('multichannel files are converted to stereo')
            audio_in = np.stack((audio_in[0], audio_in[1]), axis = 0) # channel 0 and 1 are used as the Left and Right channels.
            num_chan = (audio_in.shape[0])

        librosa.output.write_wav(os.path.join(out_data_dir_psfx, str(basefilename)), audio_in, fs)

    for file in range(0, len(predict_text_files), 1):

        # Load Texture, downsample to 44.100Hz, with an offset of 1sec., load first 60 sec.
        audio_textpredict, sfreq = librosa.load(predict_text_files[file] , sr = fs , mono = 0, offset = 3.0, duration=60.0)
        basefilename = os.path.basename(predict_text_files[file])  # store the filenames
        audio_in = librosa.util.fix_length(audio_textpredict, fs*60,axis=-1) # trim or zeropad to 60 sec.
        
        # Get the nº of channels:
        if audio_in.ndim > 1: num_chan = audio_in.shape[0]
        else: num_chan = audio_in.ndim
        print ('preparing file: ', basefilename)
        print ('num_chan: ', num_chan)
        
        # If a multichannel file is found (more than 2 channels) it is converted to stereo:
        if num_chan > 2:
            print('multichannel files are converted to stereo')
            audio_in = np.stack((audio_in[0], audio_in[1]), axis = 0) # channel 0 and 1 are used as the Left and Right channels.
            num_chan = (audio_in.shape[0])

        librosa.output.write_wav(os.path.join(out_data_dir_ptextures, str(basefilename)), audio_in, fs)


if __name__ == '__main__': pData_prep()



