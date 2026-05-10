import torch, os

single_mel = True
####################################################################################################

mel_bins = 64
batch_size = 32
epoch = 100
lr_init = 1e-3

data_space = 'Your_Dataset_Path'

endswith = '.pth'

Gesture_list_name = ['Tickle', 'Poke', 'Rub', 'Pat', 'Tap', 'Hold']
emotion_list_name = ['Happiness', 'Attention', 'Fear', 'Surprise', 'Confusion',
                                  'Sadness', 'Comfort', 'Calmimg', 'Anger', 'Disgust']

each_emotion_class_num = 1


