#!/usr/bin/env python
# coding= UTF-8
import glob
import os
import queue
import numpy as np
import librosa  #pip install librosa
import soundfile as sf  #pip install SoundFile
import sounddevice as sd    #pip install sounddevice
from tqdm import tqdm   #pip install tqdm
from joblib import Parallel, delayed
def extract_feature(file_name=None):
    if file_name: 
        #print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    else:  
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()
        def callback(i,f,t,s): q.put(i.copy())
        data = []
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True: 
                if len(data) < 100000: data.extend(q.get())
                else: break
        X = np.array(data)

    if X.ndim > 1: X = X[:,0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def extract_process(fn, label):
    try: mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
    except Exception as e:
        print("[Error] extract feature error in %s. %s" % (fn,e))
        return None
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    mu = np.mean(ext_features, axis=0)
    sigma = np.std(ext_features, axis=0)
    ext_features = (ext_features - mu)/(sigma)
    #features = np.vstack([features,ext_features])
    #labels = np.append(labels, label)
    return (ext_features, label)

def parse_audio_files(parent_dir,file_ext='*.wav'):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            print('Begin extracting', sub_dir, 'train features ...')
            partial = Parallel(n_jobs=4)(delayed(extract_process)(fn, label) for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))))
            for item in partial:
                features = np.vstack([features,item[0]])
                labels = np.append(labels, item[1])
            print("extract %s train features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)

def parse_predict_files(parent_dir,file_ext='*.wav'):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            print('Begin extracting', sub_dir, 'val features ...')
            partial = Parallel(n_jobs=4)(delayed(extract_process)(fn, label) for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))))
            for item in partial:
                features = np.vstack([features,item[0]])
                labels = np.append(labels, item[1])
            print("extract %s val features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)

def main():
    
    # Get features and labels
    features, labels = parse_audio_files('C:\\Users\\lenovo\\Desktop\\Kinetics_2\\extract\\train')
    np.save('train_feature.npy', features)
    np.save('train_label', labels)
    
    # val
    features, labels = parse_predict_files('C:\\Users\\lenovo\\Desktop\\Kinetics_2\\extract\\val')
    np.save('val_feature.npy', features)
    np.save('val_label.npy', labels)
    
if __name__ == '__main__': main()