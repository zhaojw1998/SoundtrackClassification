# Audio Classification and Audio-Video Fusion over Kinetics 600 Dataset

## 1. Introduction of the Work

This work mainly implements four jobs:

1) Extract sound tracks from Kinetics 600 Dataset;
2) Extract Melspectrograms from the sound tracks, which can be used as audio features as if processing images;
3) Audio classification over K600 Dataset;
4) Fusion with video classification results.

A high-level presentation of our work is shown in the figure below:

<img src=".vscode\\fusion.jpg" width = "65%" />

## 2. Code and File Arrangement

In this work, the codes and files are arranged as follow:

    |--root folder

        |--datalists
            Necessary directory lists (mostly in .npy format) are stored here.

        |--file_lists
            Necessary directory lists (in .txt format) are stored here. You should update K600_audio_file.txt and K600_video.txt to run this whole work on a new device. These two files list the directories of videos and sound tracks of validation set. You may write your own script for updating them, or you may adapt making_txt_list.py provided in this work.

        |--network
            Files under this folder are networks used for classification

        |--npy_files
            This folder stores audio classification outputs (in npy form).

        |--video_classification_results
            Video classification outputs (also in npy file) should be put here, which are used for video-audio fusion.

        |--scripts
            |--making_txt_list.py
                This script extract directories into an .txt file line by line. To use this script, you should replace the root directories in the code with your own directories.
            |--extract_audio_from_video.py
                This script extracts sound tracks from video files. Video folder should be arranged in this way:
                    |--K600 root folder
                        |--train
                            |--abseiling
                                |--video_1.mp4
                                --video_2.mp4
                                ……
                            |--acting_in_play
                                ……
                            ……
                        |--val
                            |--abseiling
                                |--video_1.mp4
                                --video_2.mp4
                                ……
                            |--acting_in_play
                                ……
                            ……
                The output audio files are arranged in the same way.
            |--melspectrogram_extract.py
                This script extracts melspectrograms from K600 sound tracks. Each melspectrogram is a 64X96 matrix (we call 'audio_image'), derived from a 960ms audio clip. Since we devide each sound track file (about 10s each) into several 960ms clips, one sound track file has several corresponding melspectrograms. Each melspectrogram from the train set inherits the label of the video from which it comes, and acts as a single input training sample (to increase data volumn). For the validation set, we average the output derived from the melspectrograms from the same sound track file to get final inference score (to increase inference accuracy). The output melspectrograms are arranged in the following way:
                    |--K600AudioImage
                        |--train
                            |--abseiling
                                |--audio_1.wav(This is a folder)
                                    |--audio_1_0.npy
                                    --audio_1_1.npy
                                    ……
                                |--audio_2.wav(This is a folder)
                                    |--audio_2_0.npy
                                    --audio_2_1.npy
                                    ……
                            |--acting_in_play
                                ……
                        |--val
                            …… (similar as that in the train folder)

            |--audio_clip_score_test.py
                Output of the network are melspectrogram-wise, i.e., each output score corresponds to a 960ms clip, but not the whole sound track file. This script averages the scores from corresponding melspectrograms to have a brief glance at the sound-track-wise output scores.

        |--AudioDataLoader.py
            This is the dataloader file for audio Classification

        |--file_and_label_dir.py
            This script extracts sound track directory lists (in the form of .npy) which AudioDaataLoader.py needs to use.

        |--train.py
            This is the train and validation file. 
            Note that these works were translated from a Linux platform to Windows platform, due to which certain functions were not compatible. We debuged most of the problems, but we haven't fixed one, for which tensorboardX.summaryWriter keeps throwing errors. Thus we shut it off(as you will see in the codes). To supervise the training process, we manually output train and validation accuracy in txt files instead(you'll also see relevant codes). We will try to stress this problem if time permits in the future.

        |--config.py
            Parameters such as batch size, learning rate, etc. can be modified here. Note that some parameters were designed for Video Classification and are not used in this work. It is okay to neglect them.

        |--match_video_to_audio_quantity.py
            Not all videos in K600 have sound tracks. As a result, audio output and video output have different length. Specifically, we have 29753 inference scores for videos and 29731 inference scores for audios. Both scores are stored in npy files. This script extracts the scores for sound-available videos from total video output (by forming a new npy file as video output) in order to correspond with the audio output, which facilitates the fusion process.

        |--audio_clip_score_fusion.py
            Output of the network are melspectrogram-wise, i.e., each output score corresponds to a 960ms clip, but not the whole sound track file. This script averages the scores from corresponding melspectrograms, and form a new output which is sound-track-wise.

        |--Fusion_audio_with_video.py
            This script fuses video and audio outputs. Note that both outputs should have the same length. Specifically, both outputs in our processing are 600*29731*1 npy files.

## 3. Run

To run the codes, you should:

1) extract sound track files using **extract_audio_from_video.py**;
2) extract melspectrograms using **melspectrogram_extract.py**;
3) update *K600_video_file.txt* and *K600_audio_file.txt* in the folder '*file_lists*' according to your own video and audio direcotories;
4) replace audio_image directories with your own melspectrogram folder directory. Specifically, the directories in the codes you need to update are: '*E:\\NEW_Kinetics_Audio\\Kinetics_Audio\\K600_audio_image*' and '*C:\\Users\\ZIRC2\\Desktop\\Zhao Jingwei\\K600AudioImage\\K600_audio_image*';
5) run **file_and_label_dir.py** to get necessary directory files which the dataloader require;
6) choose relevant parameters in **config.py**, and train with **train.py**;
7) run **audio_clip_score_fusion.py** to get sound-track-wise output scores;
8) put video inference output (npy file) into the folder '**video_classification_results**';
9) run **match_video_to_audio_quantity.py** to neglect scores from videos who do not have sound track;
10) run **Fusion_audio_with_video.py** to fuse audio and video scores.

If you have any problem in dealing with the codes above, please feel free to contact me.

Zhao Jingwei: zhaojw@sjtu.edu.cn

2019.12.04