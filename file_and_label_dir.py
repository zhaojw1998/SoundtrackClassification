import os
import numpy as np
import tqdm as tqdm

k600_label = {}
k600_label_file = open('C:\\Users\\ZIRC2\\Desktop\\Zhao Jingwei\\AudioRecognition\\datalists\\kinetics_600_label_list.txt')
for line in k600_label_file.readlines():
    k600_label[line.strip('\n').split('\t')[-1]] = int(line.strip('\n').split('\t')[0])
k600_label_file.close()


root = 'C:\\Users\\ZIRC2\\Desktop\\Zhao Jingwei\\K600AudioImage\\K600_audio_image\\train'
save_root = 'C:\\Users\\ZIRC2\\Desktop\\Zhao Jingwei\\AudioRecognition\\datalists'
dirs=[]
labels=[]
for label in os.listdir(root):
    label_dir = os.path.join(root, label)
    print('processing train label:', label)
    for audio in os.listdir(label_dir):
        audio_dir = os.path.join(label_dir, audio)
        for frame in os.listdir(audio_dir):
            dirs.append(os.path.join(audio_dir, frame))
            labels.append(k600_label[label])
np.save(os.path.join(save_root, 'train_audio_list.npy'), np.array(dirs))
np.save(os.path.join(save_root, 'train_label_list.npy'), np.array(labels))

root = 'C:\\Users\\ZIRC2\\Desktop\\Zhao Jingwei\\K600AudioImage\\K600_audio_image\\val'
save_root = 'C:\\Users\\ZIRC2\\Desktop\\Zhao Jingwei\\AudioRecognition\\datalists'
dirs=[]
labels=[]
for label in os.listdir(root):
    label_dir = os.path.join(root, label)
    print('processing val label:', label)
    for audio in os.listdir(label_dir):
        audio_dir = os.path.join(label_dir, audio)
        for frame in os.listdir(audio_dir):
            dirs.append(os.path.join(audio_dir, frame))
            labels.append(k600_label[label])
np.save(os.path.join(save_root, 'val_audio_list.npy'), np.array(dirs))
np.save(os.path.join(save_root, 'val_label_list.npy'), np.array(labels))
a=np.load(os.path.join(save_root, 'val_audio_list.npy'))
b=np.load(os.path.join(save_root, 'val_label_list.npy'))
print(a)
print(b)