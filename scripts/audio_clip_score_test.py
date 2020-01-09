#Output of the network is the audio-clip-wise accuracy. Here we compute the audio-file-wise accuracy, where scores of clips from the same audio file is averaged.
import numpy as np
import os

root='E:\\NEW_Kinetics_Audio\\Kinetics_Audio\\K600_audio_image\\val'
output = np.load('npy_files\\K600_audio_output_labels.npy_1014.npy')
ground_truth = np.load('npy_files\\new_fusion_groundtruth_labels_1014.npy')
accs=[]
current=0
for label in os.listdir(root):
    label_dir=os.path.join(root, label)
    volume = len(os.listdir(label_dir))

    sample = output[current: current+volume][:]
    sample_label = ground_truth[current: current+volume]
    sample_output_label = np.argmax(sample, axis = 1)
    acc = np.sum(sample_output_label == sample_label)/len(sample_label)
    accs.append(acc)
    print(acc)
    current += volume
