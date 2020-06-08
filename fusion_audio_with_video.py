import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
#from scipy.special import softmax

#fuse weights_29753 and new_weights_29753, and calculate prediction precision
weights_29753 = np.load('./npy_files/K600_audio_output_labels_1014.npy')
weights_29753 = weights_29753[:, np.newaxis, :]
new_weights_29753 = np.load('./video_classification_results/video_weights_adapt_to_audio.npy')
#print(weights_29753.shape, new_weights_29753.shape)

labels = []
with open('./datalists/kinetics_600_label_list.txt') as f:
    for item in f.readlines():
        label = item.strip('\n').split('\t')[-1]
        labels.append(label)
#print(labels)
record = [0, 0, 0, 0]
with open('file_lists/record.txt', 'wb') as f:
    pickle.dump(record, f)
    
def compute_acc(a, weights_29753, new_weights_29753):
    a = a/1000.
    weights_final = a*weights_29753 + (1-a)*new_weights_29753
    #weights_final = (softmax(weights_29753, axis=2)**a) * (softmax(new_weights_29753, axis=2)**(1-a))  #equivalent as above


    pridict = np.argmax(weights_final, axis=2)
    #print('The largest index of prediction is ' + str(pridict.max()) + '.')

    pridict_label = []
    pridict_label_5 = []
    for index in pridict.flatten().tolist():
        pridict_label.append(labels[index])
    for i in range(0, weights_29753.shape[0]):
        pridict_5 = weights_final[i].flatten().argsort()[-5:][::-1].tolist()
        pridict_5_to_label = []
        for index in pridict_5:
            pridict_5_to_label.append(labels[index])
        pridict_label_5.append(pridict_5_to_label)

    with open('file_lists/K600_audio_file.txt', 'r') as f_753:
        dir_753 = f_753.readlines()
    ground_truth = []
    for item in dir_753:
        ground_truth.append(item.strip('\n').split('\\')[-2])
    #print(str(len(ground_truth)) + ' ' + 'This number should be ' + str(weights_29753.shape[0]) +'.')

    total_score = 0
    total_score_5 = 0
    for i in range(len(pridict_label)):
        if pridict_label[i] == ground_truth[i]:
            total_score +=1
    precision = float(total_score)/len(pridict_label)
    #print('The final Top1 precision after fusion is:' + ' ' + str(precision))

    for i in range(len(pridict_label)):
        if ground_truth[i] in pridict_label_5[i]:
            total_score_5 += 1
    precision_5 = float(total_score_5)/len(pridict_label)

    with open('file_lists/record.txt', 'rb') as f:
        record = pickle.load(f)
    if (precision_5+precision)/2 > record[-1]:
        record = [a, precision, precision_5, (precision_5+precision)/2]
        with open('file_lists/record.txt', 'wb') as f:
            pickle.dump(record, f)

Parallel(n_jobs=4)(delayed(compute_acc)(a, weights_29753, new_weights_29753) for a in tqdm(range(0, 500)))

with open('file_lists/record.txt', 'rb') as f:
        record = pickle.load(f)
print('record:', record)
    

#print('The final Top5 precision after fusion is:' + ' ' + str(precision_5))
#print('The final avg precision after fusion is:' + ' ' + str((precision_5+precision)/2))
print('best record:', record)