#Output of the network is the audio-clip-wise accuracy. Here we compute the audio-file-wise Top-1 accuracy, where scores of clips from the same audio file is averaged.
import numpy as np
import os

output = np.load('./npy_files/K600_audio_output_labels_1018.npy')
ground_truth = np.load('./npy_files/new_fusion_groundtruth_labels_1018.npy')
accs=[]
current=0
for i in range(600):
    sample = output[ground_truth==i]
    sample_output_label = np.argmax(sample, axis = 1)
    acc = np.sum(sample_output_label == i)/len(sample)
    accs.append(acc*len(sample))
    #print(acc)
print('Top-1 accuracy:', np.sum(accs)/output.shape[0])
