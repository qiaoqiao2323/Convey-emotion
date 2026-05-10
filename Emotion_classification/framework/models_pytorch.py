import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config


def move_data_to_device(x, device, using_float=False):
    if isinstance(x, tuple):
        return tuple(move_data_to_device(each_x, device, using_float=using_float) for each_x in x)

    if isinstance(x, list):
        return [move_data_to_device(each_x, device, using_float=using_float) for each_x in x]

    # At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported.
    x = x.copy()

    if using_float:
        x = torch.Tensor(x)
    else:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)

        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)

        else:
            raise Exception("Error!")

    x = x.to(device)

    return x


class SingleModalModel(nn.Module):
    def __init__(self, model, modality):
        super(SingleModalModel, self).__init__()
        self.model = model
        self.modality = modality

    def forward(self, x):
        if self.modality == 'audio':
            tactile = x.new_zeros((x.shape[0], 450, 25))
            return self.model(x, tactile)

        if self.modality == 'tactile':
            audio = x.new_zeros((x.shape[0], 1001, 64))
            return self.model(audio, x)

        raise ValueError('Unknown modality: ' + str(self.modality))


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



# ----------------------------------------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


############################################################################
def init_weights(m):
 print(m)
 if type(m) == nn.Linear:
   print(m.weight)
 else:
   print('error')

class MMTM(nn.Module):
  def __init__(self, dim_visual, dim_skeleton, ratio):
    super(MMTM, self).__init__()
    dim = dim_visual + dim_skeleton
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Linear(dim, dim_out)

    self.fc_visual = nn.Linear(dim_out, dim_visual)
    self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    # initialize
    with torch.no_grad():
      self.fc_squeeze.apply(init_weights)
      self.fc_visual.apply(init_weights)
      self.fc_skeleton.apply(init_weights)

  def forward(self, visual, skeleton):
    squeeze_array = []
    for tensor in [visual, skeleton]:
      tview = tensor.view(tensor.shape[:2] + (-1,))
      squeeze_array.append(torch.mean(tview, dim=-1))
    squeeze = torch.cat(squeeze_array, 1)

    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)

    vis_out = self.fc_visual(excitation)
    sk_out = self.fc_skeleton(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(visual.shape) - len(vis_out.shape)
    vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

    dim_diff = len(skeleton.shape) - len(sk_out.shape)
    sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

    return visual * vis_out, skeleton * sk_out


#################################### mha ########################################################
import numpy as np
# transformer
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                 output_dim=d_model):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, output_dim)

    def forward(self, Q, K, V, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, output_dim=d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask(output_dim=output_dim)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim=d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(output_dim) for _ in range(n_layers)])
        self.mel_projection = nn.Linear(input_dim, d_model)

    def forward(self, enc_inputs):
        # print(enc_inputs.size())  # torch.Size([64, 54, 8, 8])
        size = enc_inputs.size()
        enc_inputs = enc_inputs.reshape(size[0], size[1], -1)
        enc_outputs = self.mel_projection(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
#################################################################################################


class CNN_Transformer(nn.Module):
    def __init__(self, class_num):

        super(CNN_Transformer, self).__init__()

        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 256
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        #########################################################################################
        out_channels = 64
        self.conv1_tactile = nn.Conv2d(in_channels=1,
                                       out_channels=out_channels,
                                       kernel_size=(3, 3), stride=(1, 1),
                                       padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2_tactile = nn.Conv2d(in_channels=64,
                                       out_channels=out_channels,
                                       kernel_size=(3, 3), stride=(1, 1),
                                       padding=(0, 0), bias=False)

        out_channels = 256
        self.conv3_tactile = nn.Conv2d(in_channels=128,
                                       out_channels=out_channels,
                                       kernel_size=(3, 1), stride=(1, 1),
                                       padding=(0, 0), bias=False)
        #########################################################################################

        d_model = 512
        self.mha = Encoder(input_dim=64, n_layers=1, output_dim=d_model)

        last_units = 256
        self.fc_fusion = nn.Linear(1024, last_units, bias=True)
        self.fc_final = nn.Linear(last_units, class_num, bias=True)

        self.mmtm = MMTM(64, 64, 4)
        self.mmtm_64_x = nn.Linear(7424, 64, bias=True)
        self.mmtm_64_tactile = nn.Linear(9216, 64, bias=True)

    def forward(self, input_a, tactile):
        # print(input.shape)
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input_a.shape

        x = input_a.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = F.relu_(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([32, 64, 249, 20])

        x = F.relu_(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([32, 128, 61, 6])

        x = F.relu_(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(2, 3))
        # print(x.size())  # torch.Size([32, 256, 28, 1])
        x = x.transpose(1, 2)  # torch.Size([64, 6, 256, 1])

        ###########
        (_, seq_len, mel_bins) = tactile.shape

        tactile = tactile.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(tactile.size())  # torch.Size([32, 1, 450, 25])
        tactile = F.relu_(self.conv1_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(2, 2))
        # print(tactile.size())  # torch.Size([32, 64, 224, 11])

        tactile = F.relu_(self.conv2_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(2, 3))
        # print(tactile.size())  # torch.Size([32, 128, 111, 3])

        tactile = F.relu_(self.conv3_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(3, 3))
        # print(tactile.size())  # torch.Size([32, 256, 36, 1])

        tactile = tactile.transpose(1, 2)  # torch.Size([32, 256, 5, 1])
        # print(tactile.size())  # torch.Size([32, 34, 256, 1])

        ###########
        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        tactile = torch.flatten(tactile, start_dim=1)
        # print(tactile.size())

        x = self.mmtm_64_x(x)
        tactile = self.mmtm_64_tactile(tactile)

        #################################### MMTM2
        x_k, tactile_k = self.mmtm(x, tactile)
        ####################################
        # print(x_k.shape, tactile_k.shape) # torch.Size([32, 64]) torch.Size([32, 64])
        x_common = torch.cat([x_k[:, None], tactile_k[:, None]], dim=1)
        # print(x_common.shape) # torch.Size([32, 2, 64])

        x, x_self_attns = self.mha(x_common)  # already have reshape
        # print(x_event.size())  # torch.Size([64, 28+34, 512])

        # 由于H_t没有变，所以这里依然 应该是： 14*64=896
        x = torch.flatten(x, start_dim=1)
        # print(x.size())  # torch.Size([32, 4160])

        x = F.relu_(self.fc_fusion(x))

        t = self.fc_final(x)

        return t




class CNN_LSTM(nn.Module):
    def __init__(self, class_num):

        super(CNN_LSTM, self).__init__()

        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 16
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        #########################################################################################
        out_channels = 64
        self.conv1_tactile = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2_tactile = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 16
        self.conv3_tactile = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        #########################################################################################

        # https://blog.csdn.net/Dog_King_/article/details/138246206
        # input_size : 也就是指的input_features，特征的维度。
        # hidden_size :隐藏层的神经元h的个数
        # num_layers: 隐藏层的层数（默认每个隐藏层里的神经元个数全都是hidden_size）
        # nonlinearity：激活函数
        # dropout：是否应用dropout, 默认不使用，如若使用将其设置成一个0-1的数字即可
        # batch_first:我们一般的数据传入都是(batch_size,num_steps,input_features)，也即是batch在第0维度，最前面，如果想这样直接扔进去进行RNN运算，batch_first需要设置为True，但是batch_first默认为False,需要我们利用permute函数，x=x.permute(1,0,2)把num_steps放在前面。
        # bidirectional：从前面的图片也可以看出，我们的RNN是单向，只能提取之前的信息，如果想提取双向的信息，这里需要把bidirectional设置为True。

        hidden_size = 64
        num_layers = 2
        # #这里注意batch_first 就是维度放在第一维的是batch_size ,这样的设置在forward里就不要变换维度了
        #         #但是需要注意batch_first = True 这个对隐藏态和细胞状态h和c的形状没有影响，其batch_size 都在第二维上

        # 也可以自己写 LSTM
        # https://wangjiosw.github.io/2020/04/20/deep-learning/rnn-pytorch/index.html
        self.LSTM = nn.LSTM(input_size=64,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional = False)


        last_units = 256
        self.fc_fusion = nn.Linear(128, last_units, bias=True)

        self.fc_final = nn.Linear(last_units, class_num, bias=True)

        self.mmtm = MMTM(64, 64, 4)
        self.mmtm_64_x = nn.Linear(464, 64, bias=True)
        self.mmtm_64_tactile = nn.Linear(576, 64, bias=True)

    def forward(self, input_a, tactile):
        # print(input.shape)  # torch.Size([32, 1000, 64])
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input_a.shape

        x = input_a.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(x.size())  #torch.Size([32, 1, 1001, 64])
        x = F.relu_(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([32, 64, 249, 20])

        x = F.relu_(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([32, 128, 61, 6])

        x = F.relu_(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(2, 3))
        # print(x.size())  # torch.Size([32, 256, 28, 1])

        x = x.transpose(1, 2)
        # print(x.size())  # torch.Size([32, 28, 256, 1])
        # # (batch, time_frames, filters), 刚好可以以相同的维度，对比下LSTM 和 Transformer 哪个好


        ####################################################################################################
        (_, seq_len, mel_bins) = tactile.shape

        tactile = tactile.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(tactile.size())  # torch.Size([32, 1, 450, 25])
        tactile = F.relu_(self.conv1_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(2, 2))
        # print(tactile.size())  # torch.Size([32, 64, 224, 11])

        tactile = F.relu_(self.conv2_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(2, 3))
        # print(tactile.size())  # torch.Size([32, 128, 111, 3])

        tactile = F.relu_(self.conv3_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(3, 3))
        # print(tactile.size())  # torch.Size([32, 256, 36, 1])

        tactile = tactile.transpose(1, 2)  # torch.Size([32, 256, 5, 1])
        # print(tactile.size())  # torch.Size([32, 34, 256, 1])

        # print(x.shape, tactile.shape)
        # original: torch.Size([32, 29, 256, 1]) torch.Size([32, 36, 256, 1])
        # 修改后： torch.Size([32, 29, 64, 1]) torch.Size([32, 36, 64, 1])

        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        # torch.Size([32, 464])
        tactile = torch.flatten(tactile, start_dim=1)
        # print(tactile.size())
        # # torch.Size([32, 576])

        x = self.mmtm_64_x(x)
        tactile = self.mmtm_64_tactile(tactile)

        #################################### MMTM2
        x_k, tactile_k = self.mmtm(x, tactile)
        ####################################
        # print(x_k.shape, tactile_k.shape) # torch.Size([32, 64]) torch.Size([32, 64])
        x_common = torch.cat([x_k[:, None], tactile_k[:, None]], dim=1)
        # print(x_common.shape) # torch.Size([32, 2, 64])
        ####################################################################################################

        x, (h_n, c_n) = self.LSTM(x_common)

        # 由于H_t没有变，所以这里依然 应该是： 14*64=896
        x = torch.flatten(x, start_dim=1)
        # print(x.size())  # torch.Size([32, 4160])

        x = F.relu_(self.fc_fusion(x))

        t = self.fc_final(x)


        return t


class CNN_GRU(nn.Module):
    def __init__(self, class_num):

        super(CNN_GRU, self).__init__()

        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 256
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        #########################################################################################
        out_channels = 64
        self.conv1_tactile = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2_tactile = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 256
        self.conv3_tactile = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        #########################################################################################

        # https://blog.csdn.net/Dog_King_/article/details/138246206
        # input_size : 也就是指的input_features，特征的维度。
        # hidden_size :隐藏层的神经元h的个数
        # num_layers: 隐藏层的层数（默认每个隐藏层里的神经元个数全都是hidden_size）
        # nonlinearity：激活函数
        # dropout：是否应用dropout, 默认不使用，如若使用将其设置成一个0-1的数字即可
        # batch_first:我们一般的数据传入都是(batch_size,num_steps,input_features)，也即是batch在第0维度，最前面，如果想这样直接扔进去进行RNN运算，batch_first需要设置为True，但是batch_first默认为False,需要我们利用permute函数，x=x.permute(1,0,2)把num_steps放在前面。
        # bidirectional：从前面的图片也可以看出，我们的RNN是单向，只能提取之前的信息，如果想提取双向的信息，这里需要把bidirectional设置为True。

        hidden_size = 64
        num_layers = 2
        # #这里注意batch_first 就是维度放在第一维的是batch_size ,这样的设置在forward里就不要变换维度了
        #         #但是需要注意batch_first = True 这个对隐藏态和细胞状态h和c的形状没有影响，其batch_size 都在第二维上

        # 也可以自己写 LSTM
        # https://wangjiosw.github.io/2020/04/20/deep-learning/rnn-pytorch/index.html
        self.gru = nn.GRU(input_size=64,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=False)

        last_units = 256
        self.fc_fusion = nn.Linear(128, last_units, bias=True)

        self.fc_final = nn.Linear(last_units, class_num, bias=True)

        self.mmtm = MMTM(64, 64, 4)

        self.mmtm_64_x = nn.Linear(7424, 64, bias=True)
        self.mmtm_64_tactile = nn.Linear(9216, 64, bias=True)

    def forward(self, input_a, tactile):
        # print(input.shape)  # torch.Size([32, 1000, 64])
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input_a.shape

        x = input_a.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(x.size())  #torch.Size([32, 1, 1001, 64])
        x = F.relu_(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([32, 64, 249, 20])

        x = F.relu_(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([32, 128, 61, 6])

        x = F.relu_(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(2, 3))
        # print(x.size())  # torch.Size([32, 256, 28, 1])

        x = x.transpose(1, 2)
        # print(x.size())  # torch.Size([32, 28, 256, 1])
        # # (batch, time_frames, filters), 刚好可以以相同的维度，对比下LSTM 和 Transformer 哪个好


        ####################################################################################################
        (_, seq_len, mel_bins) = tactile.shape

        tactile = tactile.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(tactile.size())  # torch.Size([32, 1, 450, 25])
        tactile = F.relu_(self.conv1_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(2, 2))
        # print(tactile.size())  # torch.Size([32, 64, 224, 11])

        tactile = F.relu_(self.conv2_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(2, 3))
        # print(tactile.size())  # torch.Size([32, 128, 111, 3])

        tactile = F.relu_(self.conv3_tactile(tactile))
        tactile = F.max_pool2d(tactile, kernel_size=(3, 3))
        # print(tactile.size())  # torch.Size([32, 256, 36, 1])

        tactile = tactile.transpose(1, 2)  # torch.Size([32, 256, 5, 1])
        # print(tactile.size())  # torch.Size([32, 34, 256, 1])

        # x_common = torch.cat([x[:, :, :, 0], tactile[:, :, :, 0]], dim=1)
        # # print(x_common.size())  # torch.Size([32, 28+34, 256])
        ####################################################################################################

        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        tactile = torch.flatten(tactile, start_dim=1)
        # print(tactile.size())

        x = self.mmtm_64_x(x)
        tactile = self.mmtm_64_tactile(tactile)

        #################################### MMTM2
        x_k, tactile_k = self.mmtm(x, tactile)
        ####################################
        # print(x_k.shape, tactile_k.shape) # torch.Size([32, 64]) torch.Size([32, 64])
        x_common = torch.cat([x_k[:, None], tactile_k[:, None]], dim=1)
        # print(x_common.shape) # torch.Size([32, 2, 64])
        ####################################################################################################

        x, h_n = self.gru(x_common)

        # 由于H_t没有变，所以这里依然 应该是： 14*64=896
        x = torch.flatten(x, start_dim=1)
        # print(x.size())  # torch.Size([32, 4160])

        x = F.relu_(self.fc_fusion(x))

        t = self.fc_final(x)

        return t



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)






class MTRCNN(nn.Module):
    def __init__(self, class_num_total, batchnormal=True):

        super(MTRCNN, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        frequency_num = 6
        frequency_emb_dim = 1
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1 = ConvBlock_single_layer(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, padding=(0,0), dilation=(2, 1))
        self.conv_block3 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, padding=(0,0), dilation=(3, 1))
        self.k_3_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 5
        kernel_size = (5, 5)
        self.conv_block1_kernel_5 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0,2))
        self.conv_block2_kernel_5 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(2, 1))
        self.conv_block3_kernel_5 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(3, 1))
        self.k_5_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 7
        kernel_size = (7, 7)
        self.conv_block1_kernel_7 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 3))
        self.conv_block2_kernel_7 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(2, 1))
        self.conv_block3_kernel_7 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 2), dilation=(3, 1))
        self.k_7_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        ##################################### gnn ####################################################################

        self.bn0_tactile = nn.BatchNorm2d(25)

        frequency_num = 6
        frequency_emb_dim = 1
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1_tactile = ConvBlock_single_layer(in_channels=1, out_channels=16)
        self.conv_block2_tactile = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, padding=(0, 0),
                                                           dilation=(2, 1))
        self.conv_block3_tactile = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, padding=(0, 0),
                                                           dilation=(3, 1))
        self.k_3_freq_to_1_tactile = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 5
        kernel_size = (5, 5)
        self.conv_block1_kernel_5_tactile = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size,
                                                           padding=(0, 2))
        self.conv_block2_kernel_5_tactile = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32,
                                                                    kernel_size=kernel_size,
                                                                    padding=(0, 1), dilation=(2, 1))
        self.conv_block3_kernel_5_tactile = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64,
                                                                    kernel_size=kernel_size,
                                                                    padding=(0, 1), dilation=(3, 1))
        self.k_5_freq_to_1_tactile = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 7
        kernel_size = (7, 7)
        self.conv_block1_kernel_7_tactile = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size,
                                                           padding=(0, 3))
        self.conv_block2_kernel_7_tactile = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32,
                                                                    kernel_size=kernel_size,
                                                                    padding=(0, 2), dilation=(2, 1))
        self.conv_block3_kernel_7_tactile = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64,
                                                                    kernel_size=kernel_size,
                                                                    padding=(0, 2), dilation=(3, 1))
        self.k_7_freq_to_1_tactile = nn.Linear(frequency_num, frequency_emb_dim, bias=True)
        # ----------------------------------------------------------------------------------------------------

        scene_event_embedding_dim = 128
        # embedding layers
        self.fc_embedding_event = nn.Linear(64* 2 , scene_event_embedding_dim, bias=True)
        # -----------------------------------------------------------------------------------------------------------

        # ------------------- classification layer -----------------------------------------------------------------
        # self.fc_final_arousal = nn.Linear(scene_event_embedding_dim, class_num_arousal, bias=True)
        #
        # self.fc_final_vanlence = nn.Linear(scene_event_embedding_dim, class_num_vanlence, bias=True)

        self.fc_total = nn.Linear(scene_event_embedding_dim, class_num_total, bias=True)

        self.mmtm = MMTM(64, 64, 4)
        ##############################################################################################################

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)

        init_layer(self.fc_embedding_event)

        # # classification layer -------------------------------------------------------------------------------------
        # init_layer(self.fc_final_arousal)
        # init_layer(self.fc_final_vanlence)

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input_a, input_t):
        # print(input.shape)

        # torch.Size([32, 3001, 64])
        (_, seq_len, mel_bins) = input_a.shape
        x = input_a.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''


        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        batch_x = x

        # print(x.size())  # torch.Size([32, 1, 1001, 64])  (batch, channels, frames, freqs.)
        x_k_3 = self.conv_block1(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block2(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block3(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.mean_max(x_k_3)
        x_k_3_mel = F.relu_(self.k_3_freq_to_1(x_k_3))[:, :, 0]
        # print('x_k_3_mel: ', x_k_3_mel.size())  # x_k_3_mel:  torch.Size([32, 64])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 64, 1496, 64])

        x_k_5 = self.conv_block2_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([8, 128, 740, 52])

        x_k_5 = self.conv_block3_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size(), '\n')  # torch.Size([8, 256, 358, 32])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([8, 256, 5])
        x_k_5_mel = F.relu_(self.k_5_freq_to_1(x_k_5))[:, :, 0]
        # print('x_k_5_mel: ', x_k_5_mel.size())  torch.Size([32, 64])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 64, 1494, 64])

        x_k_7 = self.conv_block2_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([8, 128, 735, 48])

        x_k_7 = self.conv_block3_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size(), '\n')  # torch.Size([8, 256, 349, 20])

        x_k_7 = self.mean_max(x_k_7)  # torch.Size([8, 256, 5])
        x_k_7_mel = F.relu_(self.k_7_freq_to_1(x_k_7))[:, :, 0]
        # print('x_k_7_mel: ', x_k_7_mel.size())  torch.Size([32, 64])

        # # kernel 9 -----------------------------------------------------------------------------------------------------

        (_, seq_len, mel_bins) = input_t.shape
        tactile = input_t.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        tactile = tactile.transpose(1, 3)
        tactile = self.bn0_tactile(tactile)
        tactile = tactile.transpose(1, 3)

        batch_tactile = tactile

        # print(tactile.size())  #  (batch, channels, frames, freqs.)
        tactile_k_3 = self.conv_block1_tactile(batch_tactile, pool_size=(2, 2), pool_type='avg')
        tactile_k_3 = F.dropout(tactile_k_3, p=0.2, training=self.training)

        tactile_k_3 = self.conv_block2_tactile(tactile_k_3, pool_size=(2, 2), pool_type='avg')
        tactile_k_3 = F.dropout(tactile_k_3, p=0.2, training=self.training)

        tactile_k_3 = self.conv_block3_tactile(tactile_k_3, pool_size=(2, 2), pool_type='avg')
        tactile_k_3 = F.dropout(tactile_k_3, p=0.2, training=self.training)
        # print('tactile_k_3: ', tactile_k_3.size())  # tactile_k_3:  torch.Size([32, 64, 52, 1])

        tactile_k_3 = self.mean_max(tactile_k_3)
        # print('tactile_k_3: ', tactile_k_3.size())  # tactile_k_3:  torch.Size([32, 64, 1])
        tactile_k_3_mel = F.relu_(tactile_k_3[:, :, 0])
        # print('tactile_k_3_mel: ', tactile_k_3_mel.size())  # tactile_k_3_mel:  torch.Size([32, 64])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        tactile_k_5 = self.conv_block1_kernel_5_tactile(batch_tactile, pool_size=(2, 2), pool_type='avg')
        tactile_k_5 = F.dropout(tactile_k_5, p=0.2, training=self.training)
        # print(tactile_k_5.size())  # torch.Size([32, 16, 223, 12])

        tactile_k_5 = self.conv_block2_kernel_5_tactile(tactile_k_5, pool_size=(2, 2), pool_type='avg')
        tactile_k_5 = F.dropout(tactile_k_5, p=0.2, training=self.training)
        # print(tactile_k_5.size())  # torch.Size([32, 32, 107, 5])

        tactile_k_5 = self.conv_block3_kernel_5_tactile(tactile_k_5, pool_size=(2, 2), pool_type='avg')
        tactile_k_5 = F.dropout(tactile_k_5, p=0.2, training=self.training)
        # print(tactile_k_5.size(), '\n')  # torch.Size([32, 64, 47, 1])

        tactile_k_5 = self.mean_max(tactile_k_5)  # torch.Size([32, 64, 1])
        tactile_k_5_mel = F.relu_(tactile_k_5[:, :, 0])
        # print('tactile_k_5_mel: ', tactile_k_5_mel.size())  # torch.Size([32, 64])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        tactile_k_7 = self.conv_block1_kernel_7_tactile(batch_tactile, pool_size=(2, 2), pool_type='avg')
        tactile_k_7 = F.dropout(tactile_k_7, p=0.2, training=self.training)
        # print(tactile_k_7.size())  # torch.Size([32, 16, 222, 12])

        tactile_k_7 = self.conv_block2_kernel_7_tactile(tactile_k_7, pool_size=(2, 2), pool_type='avg')
        tactile_k_7 = F.dropout(tactile_k_7, p=0.2, training=self.training)
        # print(tactile_k_7.size())  # torch.Size([32, 32, 105, 5])

        tactile_k_7 = self.conv_block3_kernel_7_tactile(tactile_k_7, pool_size=(2, 2), pool_type='avg')
        tactile_k_7 = F.dropout(tactile_k_7, p=0.2, training=self.training)
        # print(tactile_k_7.size(), '\n')  # torch.Size([32, 64, 43, 1])

        tactile_k_7 = self.mean_max(tactile_k_7)
        # print(tactile_k_7.size(), '\n')  # torch.Size([32, 64, 1])
        tactile_k_7_mel = F.relu_(tactile_k_7[:, :, 0])
        # print('x_k_7_mel: ', x_k_7_mel.size())  # torch.Size([32, 64])


        # -------------------------------------------------------------------------------------------------------------
        # event_embs_log_mel = torch.cat([x_k_3_mel, x_k_5_mel,
        #                                 x_k_7_mel,
        #                                 tactile_k_3_mel, tactile_k_3_mel,
        #                                 tactile_k_3_mel
        #                                 ], dim=-1)
        # print(event_embs_log_mel.size())  # torch.Size([32, 64*4])  (node_num, batch, edge_dim)

        x_k = x_k_3_mel + x_k_5_mel + x_k_7_mel
        tactile_k = tactile_k_3_mel + tactile_k_3_mel + tactile_k_3_mel
        # print(x_k.shape, tactile_k.shape)
        # # torch.Size([32, 64]) torch.Size([32, 64])
        #################################### MMTM2
        x_k, tactile_k = self.mmtm(x_k, tactile_k)
        ####################################
        # print(x_k.shape, tactile_k.shape)
        # torch.Size([32, 64]) torch.Size([32, 64])
        event_embs_log_mel = torch.cat([x_k, tactile_k], dim=-1)
        # print(event_embs_log_mel.shape)

        # -------------------------------------------------------------------------------------------------------------
        event_embeddings = F.gelu(self.fc_embedding_event(event_embs_log_mel))
        # -------------------------------------------------------------------------------------------------------------

        # arousal = self.fc_final_arousal(event_embeddings)
        #
        # vanlence = self.fc_final_vanlence(event_embeddings)

        total = self.fc_total(event_embeddings)

        return total





class PANN(nn.Module):
    def __init__(self, class_num, batchnormal=False):

        super(PANN, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)
            # self.bn0_loudness = nn.BatchNorm2d(1)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=16)

        #####################
        self.bn0_tactile = nn.BatchNorm2d(25)
        self.conv_block1_tactile = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_tactile = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_tactile = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_tactile = ConvBlock(in_channels=256, out_channels=16)

        ############################


        # # ------------------- classification layer -----------------------------------------------------------------

        self.fc_fusion = nn.Linear(128, 4, bias=True)
        self.fc_final_event = nn.Linear(128, class_num, bias=True)

        self.mmtm = MMTM(64, 64, 4)
        self.mmtm_64_x = nn.Linear(16*248, 64, bias=True)
        self.mmtm_64_tactile = nn.Linear(16*28, 64, bias=True)

        # ##############################################################################################################

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input_a, tactile):

        if config.single_mel:
            # torch.Size([32, 3001, 64])
            (_, seq_len, mel_bins) = input_a.shape
            x = input_a.view(-1, 1, seq_len, mel_bins)
            '''(samples_num, feature_maps, time_steps, freq_num)'''

        else:
            # torch.Size([32, 2, 3001, 64])
            (_, channels, seq_len, mel_bins) = input_a.shape
            x = input_a.view(-1, channels, seq_len, mel_bins)
            '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # -------------------------------------------------------------------------------------------------------------
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())
        # x.size(): torch.Size([64, 64, 1500, 32])
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size()) # x.size(): torch.Size([64, 128, 750, 16])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 256, 375, 8])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([64, 512, 187, 4])
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.flatten(x, start_dim=2)
        # print('x.size():', x.size())   #x.size(): torch.Size([32, 512, 248])
        ###################
        # torch.Size([32, 3001, 64])
        (_, seq_len, mel_bins) = tactile.shape
        tactile = tactile.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        tactile = tactile.transpose(1, 3)
        tactile = self.bn0_tactile(tactile)
        tactile = tactile.transpose(1, 3)

        tactile = self.conv_block1_tactile(tactile, pool_size=(2, 2), pool_type='avg')
        # print('tactile.size():', tactile.size())
        # tactile.size(): torch.Size([32, 64, 225, 12])
        tactile = F.dropout(tactile, p=0.2, training=self.training)

        tactile = self.conv_block2_tactile(tactile, pool_size=(2, 2), pool_type='avg')
        # print('tactile.size():', tactile.size())  # tactile.size(): torch.Size([32, 128, 112, 6])
        tactile = F.dropout(tactile, p=0.2, training=self.training)
        tactile = self.conv_block3_tactile(tactile, pool_size=(2, 2), pool_type='avg')
        # print('tactile.size():', tactile.size())  # tactile.size(): torch.Size([32, 256, 56, 3])
        tactile = F.dropout(tactile, p=0.2, training=self.training)
        tactile = self.conv_block4_tactile(tactile, pool_size=(2, 2), pool_type='avg')
        # print('tactile.size():', tactile.size())  # tactile.size(): torch.Size([32, 512, 28, 1])
        tactile = F.dropout(tactile, p=0.2, training=self.training)
        tactile = torch.flatten(tactile, start_dim=2)

        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        # torch.Size([32, 464])
        tactile = torch.flatten(tactile, start_dim=1)
        # print(tactile.size())
        x = self.mmtm_64_x(x)
        tactile = self.mmtm_64_tactile(tactile)
        # print(x.shape, tactile.shape) torch.Size([32, 512, 64]) torch.Size([32, 512, 64])
        #################################### MMTM2
        x_k, tactile_k = self.mmtm(x, tactile)
        ####################################
        # print(x_k.shape, tactile_k.shape) # torch.Size([32, 64]) torch.Size([32, 64])
        x_common = torch.cat([x, tactile], dim=-1)  # 28+248
        # print(x_common.shape) # torch.Size([32, 128])
        ###################

        # # x = F.dropout(x, p=0.5, training=self.training)
        # common_embeddings = F.relu_(self.fc_fusion(x_common))
        # # clipwise_output = torch.sigmoid(self.fc_audioset(x))
        # print('common_embeddings.size():', common_embeddings.size())
        # common_embeddings = torch.flatten(common_embeddings, start_dim=1)

        common_embeddings = x_common

        event = self.fc_final_event(common_embeddings)

        return event







