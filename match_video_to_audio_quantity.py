#In case certain videos do not have sound track, here we delete the output scores of these no-sound videos for convinience of the later audio-video score fusion.
import numpy as np

weights_29753 = np.load('./npy_files/K600_audio_output_labels_1018.npy')
weights_29796 = np.load('./video_classification_results/video_classification_demo.npy')
print(weights_29753.shape, weights_29796.shape)
weights_29753 = weights_29753[:, np.newaxis, :]
new_weights_29753 = np.zeros((weights_29753.shape), dtype = 'float32')

with open('./file_lists/Kinetics_600_vallist.txt', 'r') as f_796:
    dir_796 = f_796.readlines()
with open('./file_lists/K600_audio_file.txt', 'r') as f_753:
    dir_753 = f_753.readlines()
dir_796 = [video.strip('\n').split(' ')[0].split('/')[-1][0:20] for video in dir_796]
dir_753 = [audio.strip('.wav\n').split('\\')[-1][0:20] for audio in dir_753]
assert(len(dir_753) == weights_29753.shape[0])
i=0
j=0
while j <weights_29753.shape[0]:
    #assert(dir_753[j] in dir_796)
    if dir_753[j] in dir_796:
        new_weights_29753[j] = weights_29796[dir_796.index(dir_753[j])]
    else:
        print(dir_753[j])
        i+=1
    j+=1
print(j, i)
np.save('./video_classification_results/video_weights_adapt_to_audio.npy', new_weights_29753)