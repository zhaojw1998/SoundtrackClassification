import os
from moviepy.editor import * #install by "pip install moviepy"

root = '/data1/data/Kinetics'
target = '/data1/data/Kinetics/Kinetics_Audio'
if not os.path.exists(target):
    os.mkdir(target)

train_root = os.path.join(root, 'train')
for label in os.listdir(train_root):
    label_dir = os.path.join(train_root, label)
    audio_train = os.path.join(target, 'train')
    audio_train_label = os.path.join(audio_train, label)
    if not os.path.exists(audio_train_label):
        os.makedirs(audio_train_label)
    print('Extract ' + label + ' audios from training set')
    for video in os.listdir(label_dir):
        video_dir = os.path.join(label_dir, video)
        audio_dir = os.path.join(audio_train_label, video).strip('.mp4')+'.wav'
        if not os.path.exists(audio_dir):
            print (video_dir)
            try:
                video = VideoFileClip(video_dir)
                try:
                    video.audio.write_audiofile(audio_dir)
                except:
                    pass
                
            except:
                pass
            
            else:
                video.close()

val_root = os.path.join(root, 'val')
for label in os.listdir(val_root):
    label_dir = os.path.join(val_root, label)
    audio_val = os.path.join(target, 'val')
    audio_val_label = os.path.join(audio_val, label)
    if not os.path.exists(audio_val_label):
        os.makedirs(audio_val_label)
    print('Extract ' + label + ' audios from validation set')
    for video in os.listdir(label_dir):
        video_dir = os.path.join(label_dir, video)
        audio_dir = os.path.join(audio_val_label, video).strip('.mp4')+'.wav'
        if not os.path.exists(audio_dir):
            print (video_dir)
            try:
                video = VideoFileClip(video_dir)
                try:
                    video.audio.write_audiofile(audio_dir)
                except:
                    pass
                
            except:
                pass
            
            else:
                video.close()