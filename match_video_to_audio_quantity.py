#In case certain videos do not have sound track, here we delete the output scores of these no-sound videos for convinience of the later audio-video score fusion.
import numpy as np

large = []
small = {}

weights_small = np.load('npy_files\\K600_audio_output_labels.npy')
weights_small = weights_small[:, np.newaxis, :]
weights_large = np.load('video_classification_results\\k600_scores_1116.npy')
new_weights_large = np.zeros((weights_small.shape[0], 1 ,600), dtype = 'float32')

with open('file_lists\\K600_video_file.txt') as f:
    large_raw = f.readlines()

with open('file_lists\\K600_audio_file.txt') as f:
    small_raw = f.readlines()

for item in large_raw:
    video = item.strip('\n').split('/')[-1][0:20]
    large.append(video)

for index, item in enumerate(small_raw):
    audio = item.strip('.wav\n').split('\\')[-1][0:20]
    small[audio] = index

k=0
for index, video in enumerate(large):
    if video in small:
        new_weights_large[small[video]] = weights_large[index]
        k+=1
    else:
        print(video)

#print(k)
np.save('video_classification_results\\matched_to_audio_1116.npy', new_weights_large)