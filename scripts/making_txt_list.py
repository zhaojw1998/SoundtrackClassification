#This script was designed to make path list for videos
import os

txt_dir='D:\\ZhaoJingwei\\ComputerVision\\RelatatedPrograms\\audio_recognition\\file_lists\\K600_audio_file.txt'
video_dir='E:\\NEW_Kinetics_Audio\\Kinetics_Audio\\val'

content=[]
i=0
with open(txt_dir,'w') as f:
    for label in os.listdir(video_dir):
        label_path=os.path.join(video_dir,label)
        for audio in os.listdir(label_path):
            audio_path=os.path.join(label_path, audio)
            audio_path=audio_path+'\n'
            content.append(audio_path)
            i+=1
        print('label', label, 'done!')
    f.writelines(content)
print('totally', i, 'audios')
