import os
import shutil
root = 'D:\\Download\\Program\\ESC-50-master\\ESC-50-master\\audio'
train = 'D:\\Download\\Program\\ESC-50-master\\ESC-50-master\\audio\\train'
val = 'D:\\Download\\Program\\ESC-50-master\\ESC-50-master\\audio\\val'
"""
for label in os.listdir(root):
    label_dir = os.path.join(root, label)
    label_train_target = os.path.join(train, label)
    if not os.path.exists(label_train_target):
        os.makedirs(label_train_target)
    label_val_target = os.path.join(val, label)
    if not os.path.exists(label_val_target):
        os.makedirs(label_val_target)
    for audio in os.listdir(label_dir):
        audio_dir = os.path.join(label_dir, audio)
        if audio.split('-')[0] == str(1):
            audio_target = os.path.join(label_train_target, audio)
        else:
            audio_target = os.path.join(label_val_target, audio)
        shutil.move(audio_dir, audio_target)
"""
i=0
for video in os.listdir(root):
    set_flag = video.split('-')[0]
    label = video.split('-')[-1]
    if  not set_flag == str(1):
        train_dir = os.path.join(train, label)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        shutil.copy(os.path.join(root, video), os.path.join(train_dir, video))
    else:
        val_dir = os.path.join(val, label)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        shutil.copy(os.path.join(root, video), os.path.join(val_dir, video))
    i+=1
    if i % 10 == 0:
        print('proceeding to ', str(round(i/2000 * 100)), '%')