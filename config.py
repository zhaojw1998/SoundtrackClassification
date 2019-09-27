params = dict()
params['model_name'] = 'Neural Network'
params['dataset'] = 'Kinetics32'
# learning params
params['learning_rate'] = 1e-2
params['step'] = 500     # lr decay step
params['milestones'] = [8, 28, 58]
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['resume_epoch'] = 0
params['dropout'] = 0.8     #越大失活的越多
params['pretrained'] = False
params['pretrained_file_RGB'] = '/media/zjw/ZHAOJINGWEI/pretrain_file_conversion/p3d_pretrained_from_caffe_adapted_to_k-600.pth.tar'
params['pretrained_filr_nominal'] = 'C:/Users/ZIRC2/Desktop/Zhao Jingwei/LGD-Kinetics-RGB_epoch-0.pth.tar'

# dataset params
params['lmdb_root'] = '/home/sx/data2/zjw/'
params['epoch_num'] = 100
params['batch_size'] = 32    # single GPU
params['num_workers'] = 4

params['sample_rate'] = 1
# test model

params['model_test'] = '/home/zjw/pytorch-video-recognition/model/model_0019/P3D-Kinetics-RGB_epoch-67.pth.tar'
params['test_crops'] = 1
params['test_clips'] = 10

params['display'] = 1
params['log'] = 'log'

# check model structure
params['check_model'] = False

# generate new dataset
params['class_ratio'] = 1.0
params['video_ratio'] = 1.0
params['new_list'] = False

# No use
params['use_test'] = False
params['nTestInterval'] = 20
params['snapshot'] = 50

# server = 8
# if server==23:
#     if params['dataset']=='UCF101':
#         params['dataset_dir'] = '/data2/data/UCF101'
#     elif params['dataset']=='Kinetics':
#         params['dataset_dir'] = '/data2/qzk/dataset/Kinetics/frames'
#     else:
#         raise NameError
# elif server==27:
#     if params['dataset']=='UCF101':
#         params['dataset_dir'] = '/data1/data/UCF101'
#     elif params['dataset']=='Kinetics':
#         params['dataset_dir'] = '/data2/qzk/dataset/Kinetics/frames'
#     else:
#         raise NameError
# elif server==8:
#     assert params['dataset']=='Kinetics'
#     params['dataset_dir'] = '/data2/qzk/dataset/Kinetics/frames'
# else:
#     print ('Only 23 27 8 is allowed')
#     raise NameError

# model params