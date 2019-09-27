import os
import numpy as np
import tqdm as tqdm

root = 'D:\\ZhaoJingwei\\ComputerVision\\K32_audio_image\\train'
save_root = 'D:\\ZhaoJingwei\\ComputerVision\\RelatatedPrograms\\audio_recognition\\datalists'
dirs=[]
labels=[]
for num, label in enumerate(os.listdir(root)):
    label_dir = os.path.join(root, label)
    print('processing train label:', label)
    for audio in os.listdir(label_dir):
        audio_dir = os.path.join(label_dir, audio)
        for frame in os.listdir(audio_dir):
            dirs.append(os.path.join(audio_dir, frame))
            labels.append(num)
np.save(os.path.join(save_root, 'train_audio_list.npy'), np.array(dirs))
np.save(os.path.join(save_root, 'train_label_list.npy'), np.array(labels))

root = 'D:\\ZhaoJingwei\\ComputerVision\\K32_audio_image\\val'
save_root = 'D:\\ZhaoJingwei\\ComputerVision\\RelatatedPrograms\\audio_recognition\\datalists'
dirs=[]
labels=[]
for num, label in enumerate(os.listdir(root)):
    label_dir = os.path.join(root, label)
    print('processing val label:', label)
    for audio in os.listdir(label_dir):
        audio_dir = os.path.join(label_dir, audio)
        for frame in os.listdir(audio_dir):
            dirs.append(os.path.join(audio_dir, frame))
            labels.append(num)
np.save(os.path.join(save_root, 'val_audio_list.npy'), np.array(dirs))
np.save(os.path.join(save_root, 'val_label_list.npy'), np.array(labels))
a=np.load(os.path.join(save_root, 'val_audio_list.npy'))
b=np.load(os.path.join(save_root, 'val_label_list.npy'))
print(a)
print(b)