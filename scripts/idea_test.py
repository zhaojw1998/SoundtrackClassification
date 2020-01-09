
import glob
import os
import queue
import numpy as np
import librosa  #pip install librosa
import soundfile as sf  #pip install SoundFile
import sounddevice as sd    #pip install sounddevice
from tqdm import tqdm   #pip install tqdm
from joblib import Parallel, delayed
import cv2
import time
import sys
import matplotlib.pyplot as plt

video_name = 'pole_vault_3'
interval = [12, 22]
save_dir = 'C:\\Users\\lenovo\\Desktop\\test_video\\'

video = cv2.VideoCapture('C:\\Users\\lenovo\\Desktop\\test_video\\'+video_name+'.mp4')
sample_rate = int(video.get(cv2.CAP_PROP_FPS))
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
recipe_1 = (int(frame_height), int(frame_width)), (int(frame_height), int(frame_width))
recipe_2 = (int(frame_height/4), int(frame_width/4)), (int(frame_height/4), int(frame_width/4))
#window_size = (int(frame_height/4), int(frame_width/4))
#hop_size = (int(frame_height/4), int(frame_width/4))
window_size, hop_size = recipe_1
print('frame_rate:', sample_rate)
for h in range(int(frame_height // hop_size[0])):
    #print('h=', h)
    for w in range(int(frame_width // hop_size[1])):
        #print('w=', w)
        temporal_signal = []
        origin = (h*hop_size[0], w*hop_size[1])
        video.set(1, interval[0]*sample_rate)
        while True:
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            ret_val, image = video.read()
            if current_frame == interval[1]*sample_rate or (not ret_val):
                break
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, _, _ = cv2.split(img_yuv)
            
            if origin[0]+window_size[0] <= frame_height and origin[1]+window_size[1] <= frame_width:
                y_sample = y[origin[0]:origin[0]+window_size[0], origin[1]:origin[1]+window_size[1]]
            else:
                y_sample = np.zeros(window_size, dtype = y.dtype)
                for i_i, i in enumerate(range(origin[0], min(frame_height, origin[0]+window_size[0]))):
                    for j_i, j in enumerate(range(origin[1], min(frame_width, origin[1]+window_size[1]))):
                        y_sample[i_i, j_i] = y[i, j]
            temporal_signal.append(y_sample.mean())
        temporal_signal=np.array(temporal_signal, dtype=np.float32)
        #temporal_signal = (temporal_signal - temporal_signal.mean())/(np.max(temporal_signal) - np.min(temporal_signal))
        temporal_signal = (temporal_signal - np.min(temporal_signal))/(np.max(temporal_signal) - np.min(temporal_signal))*2 -1
        print(np.max(temporal_signal), np.min(temporal_signal))
        stft = librosa.stft(temporal_signal, n_fft=int(sample_rate+0.5), hop_length=int(int(sample_rate+0.5)/2))
        stft_magnitude, stft_phase = librosa.magphase(stft)
        #print(stft_magnitude)
        filename = save_dir + video_name + '{}_{}.png'.format(str(h), str(w))
        #plt.figure()
        #plt.imshow(stft_magnitude, aspect='auto', origin='lower')
        plt.imshow(np.log(stft_magnitude), aspect='auto', origin='lower')
        plt.show()
        #plt.imsave(filename, stft_magnitude, origin='lower', dpi=500)
