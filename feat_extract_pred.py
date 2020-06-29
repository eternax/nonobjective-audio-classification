#!/usr/bin/env python
# coding= UTF-8
# Non-objective-audio-classification Guillermo G. P.
# 20 January 2020
#
# This code has been inspired by the work “Learning Sound Event Classifiers from
# Web Audio with Noisy Labels”, arXiv preprint arXiv:1901.01189, 2019
#


import code
import glob
import os
import librosa
import numpy as np
import soundfile as sf
import argparse
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

def extract_feature(file_name=None):
    if file_name:
        print('Features extracted from :', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    
    ## Calculate Stereo Features:
    # Initialize variables:
    side_mid_ratio = 0
    LRimbalance = 0
    
    # Get the number of channels:
    if X.ndim > 1:
        num_chan = X.shape[1]
    else:
        num_chan = X.ndim
    print ('num_chan: ', num_chan)

    # If the file is not mono:
    if num_chan >= 2:
        X_t = X.T
        channelL = X_t[0]                   # Get the left channel
        channelR = X_t[1]                   # Get the right channel
        invertR = [-m for m in channelR]    # Invert the right channel
        invertR = np.array(invertR)
        
        side_signal = channelL + invertR    # Calculate the Side signal
        mid_signal = channelL + channelR    # Calculate the Mid signal
        
        side_rms =  np.mean(librosa.feature.rms(side_signal).T, axis=0) # Calculate RMS of the side signal
        mid_rms = np.mean(librosa.feature.rms(mid_signal).T, axis=0)    # Calculate RMS of the mid signal
        side_mid_ratio = side_rms/mid_rms                               # Calculate side/mid ratio
        
        rms_totalL = np.sum(librosa.feature.rms(channelL))              # Add the RMS of all the Left Channel
        rms_totalR = np.sum(librosa.feature.rms(channelR))              # Add the RMS of all the Right Channel
        
        LRimbalance = np.absolute((rms_totalR - rms_totalL)/(rms_totalL + rms_totalR))  # Calculate LR Imbalance

    ## Calculate Mono Features:
    # Prepare data:
    X = X.T
    X = np.array(librosa.core.to_mono(X))   # Convert all the files to mono file (we are done with the stereo features).
    X_norm = librosa.util.normalize(X)      # Normalize Audio

    # Trim the audio by eliminating the zeroes at the begining and the end of the file and save as a copy:
    trimmed_audio, idx = librosa.effects.trim(X_norm, top_db=90, ref=0.1, frame_length=2048, hop_length=512)
    len_samp = len(trimmed_audio)   # Calculate new length of the file.

    ## Calculate zero-crossing rates:
    # For a 50% overlapp:
    frame_length = 2048
    hop_length = 512
    
    # If a trimmed copy shorter than frame length*2 is found, use the original file and let the user know.
    if len_samp < 4096:
        trimmed_audio = X_norm
        print ('file to short for trim processing')

    ## Calculate zero-crossing rate:
    zero_crossing_rate = librosa.feature.zero_crossing_rate(trimmed_audio, frame_length, hop_length, center=True)
    # Calculate its standard deviation:
    std_zcr = np.std(zero_crossing_rate[0])
    
    ## Calculate RMS:
    rms = np.mean(librosa.feature.rms(trimmed_audio).T,axis=0)
    
    ## Calculate Spectral Centroid:
    spectral_centroid =  np.mean(librosa.feature.spectral_centroid(trimmed_audio, sr=sample_rate).T,axis=0)
    
    ## Calculate Spectral Flatness:
    flatness = np.mean(librosa.feature.spectral_flatness(trimmed_audio).T,axis=0)
    
    ## Calculate Spectral Bandwidth:
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(trimmed_audio, sr=sample_rate).T,axis=0)
    
    ## Calculate Spectral Rolloff:
    rolloff = np.mean(librosa.feature.spectral_rolloff(trimmed_audio, sr=sample_rate, roll_percent=0.9).T,axis=0)
    
    ## Calculate Energy and Energy Entropy:
    # Parameters:
    win = round(0.05*sample_rate)   # window size
    step = round(0.025*sample_rate) # step size
    stFeatures = audioFeatureExtraction.stFeatureExtraction(trimmed_audio, sample_rate, win, step)
    ## Energy:
    energy_stFeat = np.array(stFeatures[0][1])
    energy_stFeat = np.mean(energy_stFeat.T, axis= 0)
    ## Entropy:
    energy_entropy_stFeat = np.array(stFeatures[0][2])
    energy_entropy_stFeat = np.mean(energy_entropy_stFeat.T, axis= 0)
    
    ## Calculate Mean Onset Strength
    onset_strength = librosa.onset.onset_strength(trimmed_audio)
    onset_strength1 = librosa.onset.onset_strength(trimmed_audio, sr=sample_rate,aggregate=np.median,fmax=8000, n_mels=256)
    mean_onset_strength = np.mean(onset_strength1.T, axis= 0)
    
    ## Calculate number of peaks:
    # Add the rules for peak picking (see report pg. 19) by using the onset strength envelope.
    peak_pick = librosa.util.peak_pick(onset_strength1, 3, 3, 3, 5, 0.5, 10)
    num_peak = len(peak_pick)
    mean_peak_file = (num_peak/len(trimmed_audio) *100) # Calculate the mean, taking into account the length of the file.
    
    return side_mid_ratio, LRimbalance,std_zcr, rms, spectral_centroid,flatness, bandwidth, rolloff, energy_stFeat,energy_entropy_stFeat,mean_onset_strength,mean_peak_file

def parse_predict_files(parent_dir,file_ext='*.wav'): # Choose only .wav files from the directory
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0,12)), np.empty(0)  # Create an empty where to store the features and labels.
    filenames = []
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try: side_mid_ratio, LRimbalance,std_zcr, rms, spectral_centroid,flatness, bandwidth, rolloff, energy_stFeat,energy_entropy_stFeat,mean_onset_strength,mean_peak_file = extract_feature(fn)
                except Exception as e:
                    print("Error extracting features of %s. %s" % (fn,e))
                    continue
                ext_features = np.hstack([side_mid_ratio, LRimbalance,std_zcr, rms, spectral_centroid,flatness, bandwidth, rolloff,energy_stFeat,energy_entropy_stFeat,mean_onset_strength,mean_peak_file])
                print ('dim ext_feat: ', ext_features.shape)                    # Print the dimension of the features
                print ('files processed: ', features.shape[0]+1)                # Print the number of files processed.
                features = np.vstack([features,ext_features])
                filenames.append(fn)
                labels = np.append(labels, label)
            print("Done extracting %s " % (sub_dir))
    return np.array(features), np.array(filenames), np.array(labels)            # keep filenames and labels in separate npy files.

def main():

    # Predict new
    features, filenames, labels = parse_predict_files('predict')
    # Save features and labels as npy files
    np.save('predict_feat.npy', features)                                       # Save features in an npy file.
    np.save('predict_filenames.npy', filenames)                                 # Save filenames in an npy file.
    np.save('ground_truth_labels.npy', labels)                                  # Save ground truths in an npy file for further evaluation.

    #np.savetxt('predict_feat.csv', features, delimiter=',')                    # For user use only keep data in a csv, to be easily reviewed.
    #np.savetxt('ground_truth_labels.csv', labels, delimiter=',')


if __name__ == '__main__': main()
