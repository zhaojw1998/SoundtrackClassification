import glob
import os
import queue
import numpy as np
import librosa  #pip install librosa
import soundfile as sf  #pip install SoundFile
import sounddevice as sd    #pip install sounddevice
from tqdm import tqdm   #pip install tqdm
from joblib import Parallel, delayed

def processing(audio, label_path, label_save_path, sample_size, windowsize, stepsize, filefomat):
    audio_save_path = os.path.join(label_save_path, audio)
    if not os.path.exists(audio_save_path):
        os.mkdir(audio_save_path)

    clip, sample_rate = librosa.load(os.path.join(label_path, audio), sr=None)
    num_sample = int(clip.shape[0]//(sample_rate*sample_size))
    if num_sample != 0:
        for i in range(num_sample):
            sample = clip[int(i*sample_rate*sample_size): int((i+1)*sample_rate*sample_size)]
            mel_spec = librosa.feature.melspectrogram(sample, n_fft=int(sample_rate*windowsize)+1, hop_length=int(sample_rate*stepsize)+1, n_mels=64, sr=sample_rate, power=1.0)# fmin=fmin, fmax=fmax)
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            #print(mel_spec_db.shape)
            assert(mel_spec_db.shape == (64, 96))
            save_name = audio.strip(filefomat)+'_'+str(i)+'.npy'
            np.save(os.path.join(audio_save_path, save_name), mel_spec_db)

def melspectrogram_extract(phase, root_dir, sample_size, windowsize, stepsize, filefomat, save, num_jobs):
    save_path=os.path.join(save, phase)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    process_dir=os.path.join(root, phase)
    for label in os.listdir(process_dir):
        print('processing', phase, 'labels:', label)
        label_path=os.path.join(process_dir, label)
        label_save_path=os.path.join(save_path, label)
        if not os.path.exists(label_save_path):
            os.mkdir(label_save_path)
        #for audio in tqdm(os.listdir(label_path)):
        #    processing(audio, label_path, label_save_path, sample_size, windowsize, stepsize)
        Parallel(n_jobs=num_jobs)(delayed(processing)(audio, label_path, label_save_path, sample_size, windowsize, stepsize, filefomat) for audio in tqdm(os.listdir(label_path)))


if __name__ == '__main__':
    root = 'E:\\NEW_Kinetics_Audio\\Kinetics_Audio'
    save_path = 'E:\\NEW_Kinetics_Audio\\Kinetics_Audio\\K600_audio_image'

    melspectrogram_extract(phase='train', root_dir=root, sample_size=0.96, windowsize=0.025, stepsize=0.01, filefomat='.wav', save=save_path, num_jobs=8)
    melspectrogram_extract(phase='val', root_dir=root, sample_size=0.96, windowsize=0.025, stepsize=0.01, filefomat='.wav', save=save_path, num_jobs=8)

"""
file_name = 'C:\\Users\\lenovo\\Desktop\\Kinetics_2\\extract\\train\\baby_waking_up\\=-EQ3sI7NWx4_000003_000013.wav' 
#X, sample_rate = sf.read(file_name, dtype='float32')
file_name2 = 'D:\\ZhaoJingwei\\助飞\\05MOOC课程\\01 ChinaX(1)_哈佛大学\\Introduction to ChinaX.wav'
#Y, sample_rate = sf.read(file_name2, dtype='float32')
#print(X.shape)
#print(Y.shape)
clip, sample_rate = librosa.load(file_name, sr=None)

print(clip.shape)
print(sample_rate)
stft = librosa.stft(clip, n_fft=1024, hop_length=512)
print(stft.shape)

clip = clip[0:int(0.96*sample_rate)]
mel_spec = librosa.feature.melspectrogram(clip, n_fft=int(0.01*sample_rate), hop_length=int(0.01*sample_rate), n_mels=64, sr=sample_rate, power=1.0)# fmin=fmin, fmax=fmax)
mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
print(mel_spec_db.shape)

root = 'D:\\Download\\Program\\ESC-50-master\\ESC-50-master\\audio'
"""