#score fusion of audio classification and video(RGB) classification over Kinetics-600 dataset
import numpy as np
from scipy.special import softmax

#fuse audio_scores and video_scores, and calculate prediction precision
a = .2969
audio_scores = np.load('npy_files\\K600_audio_output_labels_1014.npy')
audio_scores = audio_scores[:, np.newaxis, :]
video_scores = np.load('video_classification_results\\matched_to_audio_1116.npy')
#print(audio_scores.shape, video_scores.shape)

labels = []
with open('file_lists\\kinetics_600_label_list.txt') as f:
    for item in f.readlines():
        label = item.strip('\n').split('\t')[-1]
        labels.append(label)
#print(labels)

weights_final = a*audio_scores + (1-a)*video_scores
#weights_final = (softmax(audio_scores, axis=2)**a) * (softmax(video_scores, axis=2)**(1-a))  #equivalent as above

pridict = np.argmax(weights_final, axis=2)
#print('The largest index of prediction is ' + str(pridict.max()) + '.')

pridict_label = []
pridict_label_5 = []
for index in pridict.flatten().tolist():
    pridict_label.append(labels[index])
for i in range(0, audio_scores.shape[0]):
    pridict_5 = weights_final[i].flatten().argsort()[-5:][::-1].tolist()
    pridict_5_to_label = []
    for index in pridict_5:
        pridict_5_to_label.append(labels[index])
    pridict_label_5.append(pridict_5_to_label)

with open('file_lists\\K600_audio_file.txt', 'r') as f_753:
    dir_753 = f_753.readlines()
ground_truth = []
for item in dir_753:
    ground_truth.append(item.strip('\n').split('\\')[-2])
print(str(len(ground_truth)) + ' ' + 'This number should be ' + str(audio_scores.shape[0]) +'.')

total_score = 0
total_score_5 = 0
for i in range(len(pridict_label)):
    if pridict_label[i] == ground_truth[i]:
        total_score +=1
precision = float(total_score)/len(pridict_label)
print('The final Top1 precision after fusion is:' + ' ' + str(precision))

for i in range(len(pridict_label)):
    if ground_truth[i] in pridict_label_5[i]:
        total_score_5 += 1
precision_5 = float(total_score_5)/len(pridict_label)
print('The final Top5 precision after fusion is:' + ' ' + str(precision_5))
print('The final avg precision after fusion is:' + ' ' + str((precision_5+precision)/2))