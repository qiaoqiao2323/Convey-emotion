import torch

single_mel = True
####################################################################################################

cuda = 1
training = 1
testing = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

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


