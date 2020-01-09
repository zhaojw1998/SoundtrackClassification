# This script fuses the scores derived from melspetrograms extracted from the same audio file, in order to enhance validation accuracy.
import numpy as np 
import os
outputs=np.load('npy_files\\resNet_output_Kinetics600_1014.npy')
k600_labels={}
k600_label_file = open('file_lists\\kinetics_600_label_list.txt', 'r')
for line in k600_label_file.readlines():
    k600_labels[line.strip('\n').split('\t')[-1]] = int(line.strip('\n').split('\t')[0])

current = 0
audio_num = 0
k=0
fusion_output = np.zeros((29731, 600))
fusion_labels = np.zeros(29731)
root='E:\\NEW_Kinetics_Audio\\Kinetics_Audio\\K600_audio_image\\val'
for label in os.listdir(root):
    label_dir=os.path.join(root, label)
    for audio in os.listdir(label_dir):
        audio_dir = os.path.join(label_dir, audio)
        volume = len(os.listdir(audio_dir))
        for i in range(volume):
            fusion_output[audio_num, :] += outputs[current + i]
            fusion_labels[audio_num] = k600_labels[label]
        if not len(os.listdir(audio_dir)) == 0:
            fusion_output[audio_num, :] /= len(os.listdir(audio_dir))
        else:
            print(audio_dir)
            k += 1
        audio_num += 1
        current += len(os.listdir(audio_dir))
    print('label', k600_labels[label], 'done')
assert(audio_num == 29731)
print('videos shorter than 960 ms: ', k)

output_label = np.argmax(fusion_output, axis=1)
print(output_label.shape)
print(fusion_labels.shape)
test = (output_label == fusion_labels)
print(np.sum(test)/len(test))
test5 = np.argsort(fusion_output, axis=1)
i=0
for j in range(29731):
    if fusion_labels[j] in test5[j, :][::-1][0:5]:
        i += 1
i /= 29731
print(i)
np.save('npy_files\\K600_audio_output_labels_1014.npy', fusion_output)
np.save('npy_files\\new_fusion_groundtruth_labels_1014.npy', fusion_labels)
