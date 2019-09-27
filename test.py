import numpy as np
"""
train=np.load('train_feature.npy')
val=np.load('val_feature.npy')
train_label=np.load('train_label.npy')
val_label=np.load('val_label.npy')

print(train.shape, val.shape, train_label.shape, val_label.shape)
print(train[0:10][0:10])

print(val[0:10][0:10])
print(train_label)
print(val_label)
"""
output = np.load('new_fusion_output_labels.npy')
ground_truth = np.load('new_fusion_groundtruth_labels.npy')
anchor = [0, 48, 88, 137, 180, 228, 275, 325, 372, 422, 466, 513, 551, 598, 640, 690, 737, 785, 830, 878, 922, 967, 1015, 1061, 1109, 1155, 1200, 1241, 1289, 1336, 1386, 1433, 1483]
print(output.shape)
print(ground_truth.shape)
accs=[]
for i in range(32):
    sample = output[anchor[i]:anchor[i+1]][:]
    sample_label = ground_truth[anchor[i]:anchor[i+1]]
    sample_output_label = np.argmax(sample, axis = 1)
    acc = np.sum(sample_output_label == sample_label)/len(sample_label)
    accs.append(acc)
    print(acc)
    i+=1
#print(len(accs))
#print(sum(accs)/len(accs))
#print(accs)