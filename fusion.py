import numpy as np 
import os
outputs=np.load('D:\\ZhaoJingwei\\ComputerVision\\RelatatedPrograms\\audio_recognition\\new_output_Kinetics32.npy')

anchor=np.load('datalists\\val_label_list.npy')


current = 0
audio_num = 0
k=0
fusion_output = np.zeros((1483, 32))
fusion_labels = np.zeros(1483)
root='D:\\ZhaoJingwei\\ComputerVision\\K32_audio_image\\val'
for label in os.listdir(root):
    label_dir=os.path.join(root, label)
    for audio in os.listdir(label_dir):
        audio_dir = os.path.join(label_dir, audio)
        volume = len(os.listdir(audio_dir))
        for i in range(volume):
            fusion_output[audio_num, :] += outputs[current + i]
            fusion_labels[audio_num] = anchor[current + i]
        if not len(os.listdir(audio_dir)) == 0:
            fusion_output[audio_num, :] /= len(os.listdir(audio_dir))
        else:
            print(audio_dir)
            k += 1
        audio_num += 1
        current += len(os.listdir(audio_dir))
assert(audio_num == 1483)
print(k)

output_label = np.argmax(fusion_output, axis=1)
print(output_label.shape)
print(fusion_labels.shape)
test = (output_label == fusion_labels)
print(np.sum(test)/len(test))
test5 = np.argsort(fusion_output, axis=1)
i=0
for j in range(1483):
    if fusion_labels[j] in test5[j, :][::-1][0:5]:
        i += 1
i /= 1483
print(i)
np.save('new_fusion_output_labels.npy', fusion_output)
np.save('new_fusion_groundtruth_labels.npy', fusion_labels)