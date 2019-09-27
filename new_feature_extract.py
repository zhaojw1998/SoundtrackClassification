from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
import numpy as np
import os
from tqdm import tqdm
"""
[fs, x] = audioBasicIO.readAudioFile('D:\\Download\\Program\\ESC-50-master\\ESC-50-master\\audio\\val\\1.wav\\1-34119-B-1.wav')
x = audioBasicIO.stereo2mono(x)
print(x.shape)

feature = audioFeatureExtraction.stFeatureExtraction(x, fs, 0.01*fs, 0.01*fs)
print(feature[0].shape)
feature = feature[0].reshape(1, -1)
print(feature.shape)
print(np.sum(feature == 0))
print(len(feature[0]))
print(feature)
com=np.zeros((1, 6780-6776))
feature = np.concatenate((feature, com), axis=1)
print(feature.shape)
mu = np.mean(feature, axis=1)
sigma = np.std(feature, axis=1)
print(mu, sigma)
feature = (feature - mu)/(sigma)
print(feature)
"""
def extract(root, feature_dim, num_labels):
    batch, labels = np.empty((0, feature_dim)), np.empty(0)
    for label_num, label in tqdm(enumerate(os.listdir(root)), total=num_labels):
        label_dir = os.path.join(root, label)
        for audio in os.listdir(label_dir):
            [fs, x] = audioBasicIO.readAudioFile(os.path.join(label_dir, audio))
            x = audioBasicIO.stereo2mono(x)
            feature = audioFeatureExtraction.stFeatureExtraction(x, fs, 0.1*fs, 0.1*fs)
            feature = feature[0].reshape(1, -1)
            if not len(feature[0]) == feature_dim:
                if len(feature[0]) > feature_dim:
                    feature = feature[0, 0: feature_dim]
                else:
                    complement = np.zeros((1, feature_dim - len(feature[0])))
                    feature = np.concatenate((feature, complement), axis = 1)
            mu = np.mean(feature, axis=1)
            sigma = np.std(feature, axis=1)
            feature = (feature - mu)/(sigma)
            batch = np.vstack([batch,feature])
            labels = np.append(labels, label_num)
    return np.array(batch), np.array(labels, dtype = np.int)

def main():
    # Get features and labels
    features, labels = extract('D:\\Download\\Program\\ESC-50-master\\ESC-50-master\\audio\\train', feature_dim=1700, num_labels=50)
    np.save('train_feature.npy', features)
    np.save('train_label', labels)

    # Predict new
    features, labels = extract('D:\\Download\\Program\\ESC-50-master\\ESC-50-master\\audio\\val', feature_dim=1700, num_labels=50)
    np.save('val_feature.npy', features)
    np.save('val_label.npy', labels)

if __name__ == '__main__': main()
