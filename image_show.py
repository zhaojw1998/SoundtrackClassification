from PIL import Image
import numpy as np 
import scipy.misc
import imageio

A = np.load('D:\\ZhaoJingwei\\ComputerVision\\K32_audio_image\\val\\baby_waking_up\\-0OK3nbLKgM_000012_000022.wav\\-0OK3nbLKgM_000012_000022_5.npy')
imageio.imwrite('audio_image.png', A)