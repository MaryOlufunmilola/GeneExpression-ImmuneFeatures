"""Defines the neural network, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
import math
import re
from sklearn import metrics as smetrics
import r2python
import logging
from scipy.stats import spearmanr
try:
    from past.builtins import basestring
except ImportError:  # Python 3 without the `future` package
    basestring = str
import copy
from torch.autograd.function import Function
from torch.autograd import Variable
# 3x3 convolution


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def conv1d(in_channels, out_channels, kernel_size=5, stride=2):
    padding_size = int((kernel_size - 1) / 2)
    # print(padding_size)
    # print(in_channels)
    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding_size, bias=False)

# Residual block


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, dropout_rate=0):
        super(ConvolutionBlock, self).__init__()
        self.dropout_rate = dropout_rate
        # print(in_channels)
        # print(out_channels)
        self.conv1 = conv1d(in_channels, out_channels, kernel_size, stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels, )
        # self.relu = nn.Residual(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        residual = x
        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)
        # out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.maxpool(out)
        return out


# FC block

class FullConnectedBlock(nn.Module):
    ''' Implmenent are resdual style fully connected layer"
    norm_layer could be either norm_layer or BatchNorm1d
    '''

    def __init__(self, in_channels, out_channels, dropout_rate, use_residual=True, norm_layer=None):
        super(FullConnectedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        # self.norm1 = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.norm2 = norm_layer(out_channels)
        # self.norm2 = norm_layer(out_channels, track_running_stats=False)

    def forward(self, x):
        residual = x
        out = x
        # out = self.norm1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.norm2(out)
        if self.use_residual and self.in_channels == self.out_channels:
            out += residual
        return out



# EmbeddingNet
class BasicLayers(nn.Module):
    '''
     Convolution layers followed by FC layers using params
    '''

    def __init__(self, params, block=ConvolutionBlock):
        super(BasicLayers, self).__init__()
        # create copy of params in case used in forward
        self.params = copy.deepcopy(params)
        self.in_channels = 1
        if len(self.params.kernel_sizes) == 1:
            self.params.kernel_sizes = [self.params.kernel_sizes[0]] * len(self.params.out_channels_list)
        if len(self.params.strides) == 1:
            self.strides = [self.params.strides[0]] * len(self.params.out_channels_list)
        self.output_size = self.params.input_size
        print("initial output size")
        print(self.output_size)
        self.convolution_block = self.make_layers(block, self.params.out_channels_list, kernel_sizes=self.params.kernel_sizes, strides=self.params.strides, dropout_rate=self.params.dropout_rate)
        # output_size is updated
        print("final convolution layer output size")
        if(len(self.params.out_channels_list) > 0):
            self.fc_output_size = self.output_size * self.params.out_channels_list[-1]
        else:
            self.fc_output_size = self.params.input_size

        print("initial fully connected size")
        print(self.fc_output_size)
        # self.params.FC_size_list.append(self.params.embedding_size)
        self.FC_block1 = self.make_layers_FC(
            FullConnectedBlock, self.params.FC_size_list, self.params.dropout_rate)

    def make_layers_FC(self, block, FC_size_list, dropout_rate):
        layers = []
        num_layers = len(FC_size_list)
        for i in range(0, num_layers):
            layers.append(block(self.fc_output_size, FC_size_list[
                          i], dropout_rate,
                norm_layer=self.params.norm_layer))
            self.fc_output_size = FC_size_list[i]
            print(self.fc_output_size)
        return nn.Sequential(*layers)

    def make_layers(self, block, out_channels_list, kernel_sizes, strides, dropout_rate):
        layers = []
        num_layers = len(out_channels_list)
        for i in range(0, num_layers):
            layers.append(block(self.in_channels, out_channels_list[
                          i], kernel_sizes[i], stride=strides[i], dropout_rate=dropout_rate))
            self.in_channels = out_channels_list[i]
            padding_size = (kernel_sizes[i] - 1) / 2
            convolution_stride = 1

            self.output_size = int(((self.output_size + 2 * padding_size - kernel_sizes[
                                   i]) / convolution_stride) + 1)  # convolution layer output_size
            # maxpool output_size
            self.output_size = int((
                (self.output_size - kernel_sizes[i]) / strides[i]) + 1)
            print(self.output_size)
        return nn.Sequential(*layers)

    def forward(self, x):
        # reshape numbatch * num_dim to numbatch * num_in_channel * num_dim
        out = x.view(x.size(0), 1, -1)
        out = self.convolution_block(out)
        temp = out.size()
        out = out.view(out.size(0), -1)
        out = self.FC_block1(out)
        return out


class EmbeddingNet(nn.Module):
    '''
    Encoder
    '''

    def __init__(self, params, block=ConvolutionBlock):
        super(EmbeddingNet, self).__init__()
        self.params = copy.deepcopy(params)
        if not hasattr(self.params, 'norm_layer'):
            self.params.norm_layer = nn.BatchNorm1d
        if self.params.norm_layer is None:
            self.params.norm_layer = nn.BatchNorm1d
        # tracer()
        self.basicLayers = BasicLayers(self.params)
        self.fc_output_size = self.basicLayers.fc_output_size

        self.norm = self.params.norm_layer(self.fc_output_size)
        self.fc3 = nn.Linear(self.fc_output_size, self.params.embedding_size)

    def forward(self, x):
        out = self.basicLayers(x)
        # tracer()
        # out = self.norm(out)
        out = self.fc3(out)
        return out


class outputLayer(nn.Module):
    '''
    decoder
    '''

    def __init__(self, params):
        super(outputLayer, self).__init__()
        self.params = copy.deepcopy(params)
        # change internal params to adapt encoder for internal_layers of decoder
        self.params.input_size = params.embedding_size
        self.survival_len, self.cont_len, self.bin_len = int(len(params.survival_indices) / 2), len(params.continuous_phenotype_indices), len(params.binary_phenotype_indices)

        self.params.embedding_size = self.survival_len + self.cont_len + self.bin_len
        self.params.out_channels_list = []  # no convolution layer
        self.params.FC_size_list = params.decoder_FC_size_list
        self.params.norm_layer = nn.BatchNorm1d
        self.internal_layers = EmbeddingNet(self.params)
        self.dense1_bn = nn.BatchNorm1d(self.survival_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # tracer()
        out = F.relu(x)
        out = self.internal_layers(out)
        surv_out = out[:, :self.survival_len]
        if self.survival_len > 0:
            surv_out = self.dense1_bn(surv_out)
        out = torch.cat((
            surv_out,
            out[:, self.survival_len:(self.survival_len + self.cont_len)],
            self.sigmoid(out[:, (self.survival_len + self.cont_len):])
        ), 1)
        return out


class VariationalEmbeddingNet(nn.Module):
    '''
    Variational Encoder
    adopted from https://github.com/pytorch/examples/blob/master/vae/main.py
    '''

    def __init__(self, params):
        super(VariationalEmbeddingNet, self).__init__()
        self.params = copy.deepcopy(params)
        # change internal params to adapt to have 2 * embedding_size  ## mu and sigma (first half are mu and other half is logvar)
        self.mu_size = params.embedding_size
        self.params.embedding_size = 2 * self.mu_size
        self.embedding_layers = EmbeddingNet(self.params)
        # weight by input/output size
        self.kld_norm_factor = self.params.batch_size * self.params.input_size / len(self.params.loss_fns)

    def reparameterize(self, mu, logvar):
        if self.training:
            # import ipdb
            # ipdb.set_trace()
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            ret = mu + eps * std
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld /= self.kld_norm_factor
        else:
            ret = mu
            kld = 0.
        return ret, kld

    def forward(self, x):
        # tracer()
        out = self.embedding_layers(x)
        mu, logvar = out[:, :self.mu_size], out[:, self.mu_size:]
        embed, kld = self.reparameterize(mu, logvar)
        return embed, mu, logvar, kld


class CharacterLevelCNN(nn.Module):
    '''
    adopted from https://github.com/vietnguyen91/Character-level-cnn-pytorch/blob/master/src/character_level_cnn.py
    '''

    def __init__(self, n_classes=16, input_length=20, input_dim=20,
                 n_conv_filters=8,
                 n_fc_neurons=64):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3, stride=1))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=6, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3, stride=1))
        # self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        # self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        # self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3, stride=1))

        # dimension = int((input_length - 96) / 27 * n_conv_filters)
        dimension = int((input_length - 19) * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        # if n_conv_filters == 256 and n_fc_neurons == 1024:
        #     self._create_weights(mean=0.0, std=0.05)
        # elif n_conv_filters == 1024 and n_fc_neurons == 2048:
        #     self._create_weights(mean=0.0, std=0.02)

        if n_conv_filters > 256 and n_fc_neurons > 1024:
            self._create_weights(mean=0.0, std=0.02)
        else:
            self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        # output = self.conv3(output)
        # output = self.conv4(output)
        # output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


class CharacterEmbeddingNet(nn.Module):
    '''
    Variational Encoder
    adopted from https://github.com/pytorch/examples/blob/master/vae/main.py
    '''

    def __init__(self, params):
        super(CharacterEmbeddingNet, self).__init__()
        self.params = copy.deepcopy(params)
        # change internal params to adapt to have 2 * embedding_size  ## mu and sigma (first half are mu and other half is logvar)
        self.input_dim = self.params.input_dim
        self.input_length = int(self.params.input_size / self.input_dim)
        try:
            self.params.n_char_conv_filters = self.params.n_char_conv_filters
        except:
            self.params.n_char_conv_filters = 32
        try:
            self.params.n_char_fc_neurons = self.params.n_char_fc_neurons
        except:
            self.params.n_char_fc_neurons = 32

        self.embedding_layers = CharacterLevelCNN(n_classes=self.params.embedding_size,
                                                  input_dim=self.input_dim, input_length=self.input_length,
                                                  n_conv_filters=self.params.n_char_conv_filters,
                                                  n_fc_neurons=self.params.n_char_fc_neurons)

    def forward(self, x):
        out = x.view(x.size(0), self.input_length, self.input_dim)
        out = self.embedding_layers(out)
        return out


class AttentionEncoder(nn.Module):
    "Attention Encoder for deepImmune produce a matrix that  could be multiplied with embedding"

    def __init__(self, params):
        super(AttentionEncoder, self).__init__()
        self.params = copy.deepcopy(params)
        if hasattr(params, 'attention_output_size'):
            self.output_size = params.attention_output_size
        else:
            self.output_size = params.embedding_size
        self.embedding_size = params.embedding_size
        # change internal params to adapt encoder for internal_layers of decoder
        self.params.input_size = params.attention_input_size

        self.params.embedding_size = self.output_size * self.embedding_size
        self.params.out_channels_list = []  # no convolution layer
        self.params.FC_size_list = params.attention_FC_size_list
        self.params.norm_layer = nn.LayerNorm
        self.internal_layers = BasicLayers(self.params)
        self.norm = self.params.norm_layer(self.internal_layers.fc_output_size)
        self.last_linear = nn.Linear(self.internal_layers.fc_output_size, self.params.embedding_size)

        # self.softmaxs = clones(nn.Softmax(dim=1), self.output_size)

    def forward(self, x):
        out = self.internal_layers(x)
        # tracer()
        out = self.last_linear(out)

        bs = out.size(0)
        out = out.view(bs, self.embedding_size, self.output_size)
        # out_list = [nn.Softmax(dim=0)(out[inx, :, :])
        #             for inx in range(bs)]
        # out = torch.stack(out_list)
        # out = out - 0.5  # for range [-0.5 0.5]
        return out


def feature_attention(attn_mat, embedding, dropout=None):
    "perform matrix multiplication "

    # import ipdb
    # ipdb.set_trace()
    bs = attn_mat.size(0)
    embedding = embedding.view([bs, 1, embedding.shape[1]])
    # note matmul and bmm are similar -- > torch.bmm(xx,yy) = torch.matmul(xx,yy)
    output = torch.bmm(embedding, attn_mat)
    output = output.view([bs, output.shape[2]])
    return output


class outputLayer_simple(nn.Module):

    def __init__(self, params):
        super(outputLayer_simple, self).__init__()
        self.linear_output_size = linear_output_size
        self.binary_output_size = binary_output_size
        self.linear1 = nn.Linear(embedding_size, self.linear_output_size + self.binary_output_size)
        self.dense1_bn = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()

    def forward(self, x):
        linear1_out = self.linear1(x)
        survival_out = self.dense1_bn(linear1_out[:, 0:1])
        binary_output = self.sigmoid(linear1_out[:, self.linear_output_size:])
        out = torch.cat((survival_out, linear1_out[:, 1:self.linear_output_size], binary_output), 1)

        return out


class tempNet(nn.Module):

    def __init__(self, block, input_size, out_channels_list=[32, 32, 32],
                 embedding_size=32, blocks=1, kernel_sizes=[5], strides=[2]):
        super(tempNet, self).__init__()
        self.in_channels = 1
        self.block = block
        if len(kernel_sizes) == 1:
            self.kernel_sizes = [kernel_sizes] * len(out_channels_list)
        if len(strides) == 1:
            self.strides = [strides] * len(out_channels_list)

        self.lay1 = self.make_layer(block, out_channels_list[
                                    0], kernel_sizes=self.kernel_sizes, strides=self.strides[0])
        # self.embedding_size = embedding_size
        # self.blocks = blocks
        # self.kernel_size = kernel_size
        # self.strides = strides
       # (1, 64, 5, stride=2

    def make_layer(self, block, out_channels, kernel_sizes, strides):
        layers = []
        layers.append(block(self.in_channels, out_channels,
                            kernel_sizes[0], stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(
                block(out_channels, out_channels, kernel_size, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        # out = self.make_layer(self.block,  self.out_channels_list[0], kernel_size = self.kernel_size, stride = self.strides[0], blocks=self.blocks)(out)
        out = self.lay1(out)
        print("here")
        print(out)
        return out


# EmbeddingNet
class EmbeddingNet_FC(nn.Module):

    def __init__(self, block, input_size, out_channels_list,
                 embedding_size=32,
                 dropout_rate=0.1):
        super(EmbeddingNet_FC, self).__init__()
        self.in_channels = input_size
        self.output_size = input_size
        self.dropout_rate = dropout_rate
        print("initial output size")
        print(self.output_size)
        self.layers_block1 = self.make_layers_FC(
            block, out_channels_list, dropout_rate)
        # output_size is updated
        print("final output size")
        print(self.output_size)
        self.fc3 = nn.Linear(self.in_channels, embedding_size)

    def make_layers_FC(self, block, out_channels_list, dropout_rate):
        layers = []
        num_layers = len(out_channels_list)
        for i in range(0, num_layers):
            layers.append(block(self.in_channels, out_channels_list[
                          i], dropout_rate))
            self.in_channels = out_channels_list[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        # reshape numbatch * num_dim to numbatch * num_in_channel * num_dim
        # out = x.view(x.size(0), 1, -1)
        out = x
        out = self.layers_block1(out)
        out = self.fc3(out)
        return out


class embedding_conv(nn.Module):

    def __init__(self, input_size, num_classes=1):
        super(ConvNet1D, self).__init__()
        layer1_size = math.floor(input_size / 2)
        output_size = math.floor(layer1_size / 2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        # self.output_size = math_ceiling
        self.fc = nn.Linear(output_size * 32, num_classes)
        self.dense1_bn = nn.BatchNorm1d(num_classes)


class oneLayerNet(nn.Module):

    def __init__(self, input_size, num_classes):
        super(oneLayerNet, self).__init__()

    def forward(self, x):
        out = nn.Linear(self.input_size, self.num_classes)(x)
        out = nn.ReLU()(out)
        out = nn.BatchNorm1d(self.num_classes)(out)
        return out


class ImmuneEmbed(nn.Module):

    def __init__(self, input_size, hidden_layers):
        super(ImmuneEmbed, self).__init__()

    def forward(self, x):
        out = nn.Linear(self.input_size, self.hidden_layer[0])(x)
        out = nn.ReLU()(out)
        out = nn.BatchNorm1d(self.hidden_layer[0])(out)
        for layer_size in self.hidden_layers:
            out = nn.Linear(layer_size)(out)
            out = nn.ReLU()(out)
            out = nn.BatchNorm1d(layer_size)(out)
        return out


# Fully connected neural network with one hidden layer

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes=1):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # batchnorm takes both channels or hidden size as input
        self.dense1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dense2_bn = nn.BatchNorm1d(num_classes)
        # self.dense2_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dense1_bn(out)
        out = self.fc2(out)
        # print(out.shape)
        out = self.dense2_bn(out)
        return out

# 1D Convolutional neural network (two convolutional layers)


class ConvNet1D(nn.Module):

    def __init__(self, input_size, num_classes=1):
        super(ConvNet1D, self).__init__()
        layer1_size = math.floor(input_size / 2)
        output_size = math.floor(layer1_size / 2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        # self.output_size = math_ceiling
        self.fc = nn.Linear(output_size * 32, num_classes)
        self.dense1_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        gene_len, batch_size = x.size()  # x is gene_len x batch_size
        # print(x.shape)
        # Turn (gene_len x batch_size) into (batch_size x input_size x gene_len) for CNN, where input_size = 1
        # x = x.transpose(0, 1)
        # print(x.shape)
        x = x.reshape(batch_size, 1, gene_len)
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.dense1_bn(out)
        return out

# Convolutional neural network (two convolutional layers)


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels]) / num_examples


def checkNaNInf(x):
    torch.isnan(x).any() or x.eq(
        float('inf')).any() or x.eq(float('-inf')).any()


def isfinite(x):
    """
    Quick pytorch test that there are no nan's or infs.

    note: torch now has torch.isnan
    url: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519
    """
    not_inf = ((x + 1) != x)
    not_nan = (x == x)
    return not_inf & not_nan

firstelem = lambda x: x[0]


def isfinite_list(x):
    """
    Quick pytorch test that there are no nan's or infs.

    note: torch now has torch.isnan
    url: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519
    """
    out = 1
    if x is not None:
        not_inf = ((x + 1) != x)
        not_nan = (x == x)
        out = not_inf & not_nan
        out = out.all()
    return out


def negative_log_partial_likelihood(survival, risk, debug=False):
    """Return the negative log-partial likelihood of the prediction
    y_true contains the survival time
    risk is the risk output from the neural network
    censor is the vector of inputs that are censored
    censor data: 1 - dead, 0 - censor
    regularization is the regularization constant (not used currently in model)

    Uses the torch backend to perform calculations

    Sorts the surv_time by sorted reverse time
    https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/deep_surv.py
    """

    # calculate negative log likelihood from estimated risk\

    # temp = survival
    # temp[:, 1] = risk

    # print(torch.stack([survival[:, 0], risk]))
    _, idx = torch.sort(survival[:, 0], descending=True)
    censor = survival[idx, 1]
    risk = risk[idx]
    # print(temp[idx, :])
    # print(survival[idx, :])
    epsilon = 0.00001
    max_value = 10
    alpha = 0.1
    risk = torch.reshape(risk, [-1])  # flatten
    # risk = risk - torch.mean(risk)
    # risk[risk > max_value] = max_value
    hazard_ratio = torch.exp(risk)

    # cumsum on sorted surv time accounts for concordance
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + epsilon)
    log_risk = torch.reshape(log_risk, [-1])
    uncensored_likelihood = risk - log_risk

    # apply censor mask: 1 - dead, 0 - censor
    censored_likelihood = uncensored_likelihood * censor
    num_observed_events = torch.sum(censor)
    neg_likelihood = - torch.sum(censored_likelihood) / \
        num_observed_events
    # if(not isfinite(neg_likelihood) or debug):
    #     import ipdb
    #     ipdb.set_trace()

    # print(type(neg_likelihood))

    # if (neg_likelihood != neg_likelihood):
    #     print(neg_likelihood)
    #     print(censor[np.isnan(censor)])
    #     print(risk[np.isnan(risk)])
    #     raise ValueError("nan found")

    # return torch.exp(neg_likelihood)
    return neg_likelihood


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


def negative_log_partial_likelihood_loss(risk, censor, debug=False):
    '''
    function to change the order of censor and risk
    '''

    return negative_log_partial_likelihood(censor, risk, debug)


def c_index(predicted_risk, survival):
    if survival is None:
        return 0
    # calculate the concordance index
    ci = np.nan  # just to know that concordance index cannot be estimated
    # print(r2python.cbind(np.reshape(predicted_risk, (-1, 1)), survival))

    try:
        na_inx = ~(np.isnan(survival[:, 0]) | np.isnan(survival[:, 1]) | np.isnan(predicted_risk))
        predicted_risk, survival = predicted_risk[na_inx], survival[na_inx]
        if len(predicted_risk) > 0 and sum(survival[:, 1] == 1) > 2:
            survival_time, censor = survival[:, 0], survival[:, 1]
            epsilon = 0.001
            partial_hazard = np.exp(-(predicted_risk + epsilon))
            censor = censor.astype(int)
            ci = concordance_index(survival_time, partial_hazard, censor)

    except:
        ci = np.nan

    return ci


# maintain all metrics required in this dictionary- these are used in the
# training and evaluation loops
def calculate_auc(pred, y):
    if y is None:
        return 0
    # print(y)
    # print("pred")
    # print(pred)
    na_inx = ~(isnan(pred) | isnan(y))
    pred, y = pred[na_inx], y[na_inx]
    try:
        fpr, tpr, thresholds = smetrics.roc_curve(y, pred, pos_label=1)
        auc = smetrics.auc(fpr, tpr)
    except:
        auc = 0.5
    if isnan(auc):
        auc = 0.5
    return auc


def spearman_corr(pred, y):
    # import ipdb
    # ipdb.set_trace()
    try:
        out = spearmanr(pred, y, nan_policy='omit')
        out = out.correlation
    except:
        out = 0.

    return out


metrics = {
    'auc': calculate_auc,
    'c_index': c_index,
    'correlation': spearman_corr
    # could add more metrics such as accuracy for each token type
}


def isnan(x):
    return x != x


def max_na(xx):
    return (0 if len(xx) == 0 else max(xx))


def mean_na(xx):
    xx = np.array(xx)
    return np.mean(xx[~np.isnan(xx)])


class Clamp(Function):
    """
    clipped outside (-clip, clip)
    https://github.com/pytorch/pytorch/blob/53fe804322640653d2dddaed394838b868ce9a26/torch/autograd/_functions/pointwise.py#L95
    """

    @staticmethod
    def forward(ctx, i, min_val, max_val):
        ctx._mask = (i.ge(min_val) * i.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        return grad_output * mask, None, None


class Clip(Function):
    """
    clipped inside (-clip, clip)
    https://github.com/pytorch/pytorch/blob/53fe804322640653d2dddaed394838b868ce9a26/torch/autograd/_functions/pointwise.py#L95
    """

    @staticmethod
    def forward(ctx, i, min_val, max_val):
        ctx._mask = (i.le(min_val) + i.ge(max_val))
        return torch.mul(i, ctx._mask.float())

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        return grad_output * mask, None, None


class MSELossClip(nn.Module):
    """MSELoss with loss clipped between {-clip, clip}
    this way clip and clamp are not evaluated at every foward but only evaluated.
    """

    def __init__(self, clip=1.0):
        super(MSELossClip, self).__init__()
        self.clip = clip
        if clip < 0:
            self.min_val = self.clip
            self.max_val = -self.clip
            self.ClipFn = Clip.apply
        else:
            self.min_val = -self.clip
            self.max_val = self.clip
            self.ClipFn = Clamp.apply

    def forward(self, outputs, labels):
        error = outputs - labels
        error = (self.ClipFn(error, self.min_val, self.max_val))**2
        # error = torch.clamp(error, min=-self.clip, max=self.clip)**2
        error = error.sum()
        return error


def make_continuous_loss(spec):
    """Build the continuous-trait loss from a config string without eval().

    Accepts "nn.MSELoss()", "MSELoss", "nn.L1Loss()", "nn.SmoothL1Loss()",
    or "MSELossClip(<number>)" / "MSELossClip(clip=<number>)".
    """
    if not isinstance(spec, str):
        return spec
    text = spec.replace(" ", "")
    simple = {"nn.MSELoss()": nn.MSELoss, "MSELoss": nn.MSELoss, "nn.MSELoss": nn.MSELoss,
              "nn.L1Loss()": nn.L1Loss, "nn.SmoothL1Loss()": nn.SmoothL1Loss}
    if text in simple:
        return simple[text]()
    m = re.fullmatch(r"(?:net\.)?MSELossClip\((?:clip=)?([-+0-9.eE]+)\)", text)
    if m:
        return MSELossClip(float(m.group(1)))
    raise ValueError("Unsupported continuous_loss in params: {!r}".format(spec))


def create_lossfns_mask(params):
    continuous_loss = nn.MSELoss()
    if hasattr(params, 'continuous_loss'):
        continuous_loss = make_continuous_loss(params.continuous_loss)

    survival_output_size = int(len(params.survival_indices) / 2)
    linear_output_size = survival_output_size + len(params.continuous_phenotype_indices)
    binary_output_size = len(params.binary_phenotype_indices)
    params.loss_fns = [negative_log_partial_likelihood_loss] * survival_output_size + \
        [continuous_loss] * len(params.continuous_phenotype_indices) + \
        [nn.BCELoss()] * len(params.binary_phenotype_indices)

    params.mask = np.concatenate([params.survival_indices, params.continuous_phenotype_indices, params.binary_phenotype_indices])
    # print(params.loss_fns)

    return params


def is_cox_loss(loss_fn):
    """True if loss_fn is the Cox partial-likelihood loss."""
    return getattr(loss_fn, '__name__', None) == 'negative_log_partial_likelihood_loss'


def regularized_loss(model, params, index="all"):
    """L1/L2 penalty on the weights of nn.Linear layers.

    index == "all": penalize every Linear weight matrix in `model`.
    index == i:     penalize only row i of the final Linear layer, i.e. the
                    weights feeding output i (one task head).

    Previously the parameter filter required names ending in both "weight"
    and "linear", which no parameter satisfies, so no penalty was ever added.
    """
    l1 = getattr(params, 'l1_regularizer', 0.) or 0.
    l2 = getattr(params, 'l2_regularizer', 0.) or 0.
    if l1 == 0 and l2 == 0:
        return 0.

    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) == 0:
        return 0.

    if index == "all":
        weights = [m.weight.view(-1) for m in linear_layers]
    else:
        final = linear_layers[-1]
        if index >= final.weight.shape[0]:
            return 0.
        weights = [final.weight[index].view(-1)]

    loss_curr = 0.
    for w in weights:
        loss_curr = loss_curr + torch.norm(w, 1) * l1 + torch.norm(w, 2) * l2
    return loss_curr


def calculate_loss(labels, net_outputs, loss_fns):
    '''
    define loss function :
    '''

    # total_loss = torch.zeros(1)
    total_loss = 0.

    # ## survival output
    # survival = labels[:,0:2]
    # na_inx = ~( np.isnan(survival[:,1]) | nbasep.isnan(survival[:,1]))
    # survival, net_output = survival[na_inx,:], net_outputs[na_inx,0]
    # if(len(label) > 1 ):
    #     loss_curr = loss_fns[0](net_output, label)
    # else:
    #     loss_curr = 0

    # total_loss = total_loss + loss_curr

    # label, net_output, loss_fn = labels[:,i], net_outputs[:, i], loss_fns[i]

    len_fns = len(loss_fns)
    label_inx = 0

    for i in range(len_fns):
        net_output, loss_fn = net_outputs[:, i], loss_fns[i]
        # print(loss_fn)
        # print(is_cox_loss(loss_fn))
        if is_cox_loss(loss_fn):
            label = labels[:, label_inx:(label_inx + 2)]
            label_inx = label_inx + 2
            # print(label.shape)
            na_inx = ~(isnan(label[:, 0]) | isnan(label[:, 1]))
            label, net_output = label[na_inx, :], net_output[na_inx]
            # print(label)
        else:
            label = labels[:, label_inx]
            label_inx = label_inx + 1
            na_inx = ~isnan(label)
            label, net_output = label[na_inx], net_output[na_inx]

        if(len(label) > 1):
            # in case of survival label is censor; censored data assumed to
            # sorted based on patient event time.
            loss_curr = loss_fn(net_output, label)
        else:
            # loss_curr = torch.zeros(1)
            loss_curr = 0.
        # print(loss_curr.item())
        # manually tries to value response rate
        # if i >= len_fns - 2:
            # loss_curr = loss_curr * 10

        total_loss = total_loss + loss_curr

    return total_loss


def define_metrics(params):
    '''
    Create the metrics if not defined
    Format  ['name of. variable', 'metrics', 'index of predicted data', 'index of real data'. ]
    '''

    survival_output_size = int(len(params.survival_indices) / 2)
    if isinstance(params.metrics[0], basestring):
        params.metrics = range(len(params.loss_fns))

    label_inx = 0
    if isinstance(params.metrics[0], int):
        metrics = []
        metrics_type = ["c_index"] * survival_output_size + \
            ["correlation"] * len(params.continuous_phenotype_indices) + \
            ["auc"] * len(params.binary_phenotype_indices)
        for i in range(len(metrics_type)):
            # i is output_batch
            loss_fn = params.loss_fns[i]
            metrics.append([params.header[label_inx] + "_" + str(i), metrics_type[i], i, label_inx])
            if is_cox_loss(loss_fn):
                label_inx = label_inx + 2
            else:
                label_inx = label_inx + 1
        params.metrics = [metrics[tt] for tt in params.metrics]

    if isinstance(params.best_model_metric, int):
        metrics_curr = params.metrics[params.best_model_metric]
        params.best_model_metric = metrics_curr[0] + "_" + metrics_curr[1]

    # tracer()

    return params


def process_negative_loss_excluded(params):
    out = params.loss_excluded_from_training
    if all(np.array(params.loss_excluded_from_training) < 0):
        out1 = [-(xx + 1) for xx in params.loss_excluded_from_training]
        out = list(set(range(len(params.loss_fns))) - set(out1))
        logging.warning("loss_excluded_from_training {}".format(str(out)))
    return out


def initializeMTL(params):
    '''
    initialize MTL specific parameters
    '''
    if not hasattr(params, 'mtl'):
        params.mtl = False
    else:
        is_regression = [torch.tensor(True)]
        is_regression += [torch.tensor(True) if isinstance(xx, (nn.MSELoss, MSELossClip)) else torch.tensor(False) for xx in params.loss_fns]
        is_regression = torch.stack(is_regression)  # True: Regression/MeanSquaredErrorLoss, False: Classification/CrossEntropyLoss/Cox-hazard model
        params.mtlloss = MultiTaskLoss(is_regression=is_regression, reduction='sum')

    return params


class MultiTaskLoss(torch.nn.Module):
    '''https://arxiv.org/abs/1705.07115
    https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
    '''

    def __init__(self, is_regression, reduction='none'):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses, mask):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars)**(1 / 2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ((self.is_regression + 1) * (stds**2))
        multi_task_losses = (coeffs * losses + torch.log(stds)) * mask

        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses


def update_loss_parameters(labels, net_outputs, embedding_model, outputs, embedding_optimizer, outputs_optimizer, params, train_optimizer_mask=[1, 1], kld=0.):
    '''
    define loss function and update parameters
    '''

    def update_parameters(loss, train_optimizer_mask, embedding_model, outputs):
        # clear previous gradients, compute gradients of all variables wrt loss
        # tracer()
        embedding_optimizer.zero_grad()
        outputs_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # xx = [p.grad for p in outputs.parameters()]
        # print(xx)
        # tracer()

        # if not all([isfinite(p.grad).all() for p in outputs.parameters()]):
        # tracer()
        # performs updates using calculated gradients
        if train_optimizer_mask[0]:
            embedding_optimizer.step()
        if train_optimizer_mask[1]:
            outputs_optimizer.step()

    device = net_outputs.device
    if params.mtl:
        total_loss = []
        total_loss.append(torch.tensor(kld, device=device))
        mtlmask = []
        mtlmask.append(torch.tensor(0., device=device) if kld == 0 else torch.tensor(1., device=device))
    else:
        total_loss = kld

    loss_fns = params.loss_fns
    len_fns = len(loss_fns)

    label_inx = 0
    if sum(train_optimizer_mask):
        is_train = True
    else:
        is_train = False

    # loss_for_training = list(set(range(len(loss_fns))) - set(params.loss_excluded_from_training))
    # import ipdb
    # ipdb.set_trace()
    for i in range(len(loss_fns)):
        net_output, loss_fn = net_outputs[:, i], loss_fns[i]
        if is_cox_loss(loss_fn):
            label = labels[:, label_inx:(label_inx + 2)]
            label_inx = label_inx + 2
            # print(label.shape)
            na_inx = ~(isnan(label[:, 0]) | isnan(label[:, 1]))
            label, net_output = label[na_inx, :], net_output[na_inx]
        else:
            label = labels[:, label_inx]
            label_inx = label_inx + 1
            na_inx = ~isnan(label)
            label, net_output = label[na_inx], net_output[na_inx]

        if(len(label) > 1) and not any(np.array(params.loss_excluded_from_training) == i):
            # in case of survival label is censor; censored data assumed to
            # sorted based on patient event time.
            loss_curr = loss_fn(net_output, label)
            loss_curr = loss_curr + regularized_loss(outputs, params, i)
        else:
            # loss_curr = torch.zeros(1)
            loss_curr = torch.tensor(0., device=device)

        if isnan(loss_curr):
            loss_curr = torch.tensor(0., device=device)

        # print(net_output)
        # print(loss_curr)
        # tracer()

        if is_train and params.pipeline_optimization and loss_curr != 0.:
            update_parameters(loss_curr, train_optimizer_mask, embedding_model, outputs)

        if params.mtl:
            mtlmask.append(torch.tensor(0., device=device) if loss_curr == 0. else torch.tensor(1., device=device))
            total_loss.append(loss_curr)
        else:
            total_loss = total_loss + loss_curr

    # tracer()
    if is_train and params.pipeline_optimization and kld != 0.:
        update_parameters(kld, train_optimizer_mask, embedding_model, outputs)

    if params.mtl:
        total_loss = params.mtlloss(torch.stack(total_loss), torch.stack(mtlmask))

    if is_train and not params.pipeline_optimization:
        update_parameters(total_loss, train_optimizer_mask, embedding_model, outputs)

    if total_loss != 0:
        loss_val = total_loss.item()
    else:
        loss_val = 0.

    return loss_val


def update_loss_parameters_vectorized(labels, net_outputs, models, optimizers, params, train_optimizer_mask):
    '''
    define loss function and update parameters
    '''

    def update_parameters(loss):
        "perorm parameter updates"
        # clear previous gradients, compute gradients of all variables wrt loss
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()
        # performs updates using calculated gradients
        for mask, optimizer in zip(train_optimizer_mask, optimizers):
            if mask:
                optimizer.step()

    total_loss = 0.
    loss_fns = params.loss_fns
    len_fns = len(loss_fns)

    label_inx = 0
    if sum(train_optimizer_mask):
        is_train = True
    else:
        is_train = False

    # loss_for_training = list(set(range(len(loss_fns))) - set(params.loss_excluded_from_training))
    # import ipdb
    # ipdb.set_trace()
    for i in range(len(loss_fns)):
        net_output, loss_fn = net_outputs[:, i], loss_fns[i]
        if is_cox_loss(loss_fn):
            label = labels[:, label_inx:(label_inx + 2)]
            label_inx = label_inx + 2
            # print(label.shape)
            na_inx = ~(isnan(label[:, 0]) | isnan(label[:, 1]))
            label, net_output = label[na_inx, :], net_output[na_inx]
        else:
            label = labels[:, label_inx]
            label_inx = label_inx + 1
            na_inx = ~isnan(label)
            label, net_output = label[na_inx], net_output[na_inx]

        if(len(label) > 1) and not any(np.array(params.loss_excluded_from_training) == i):
            # in case of survival label is censor; censored data assumed to
            # sorted based on patient event time.
            loss_curr = loss_fn(net_output, label)
            loss_curr = loss_curr + regularized_loss(models[2], params, i)
        else:
            # loss_curr = torch.zeros(1)
            loss_curr = 0.

        # if i == 2:
        #     import ipdb
        #     ipdb.set_trace()
        #     print("loss_curr")
        #     print(loss_curr)

        if isnan(loss_curr):
            loss_curr = 0.

        # print(net_output)
        # print(loss_curr)
        # tracer()

        if is_train and params.pipeline_optimization and loss_curr != 0.:
            update_parameters(loss_curr)

        total_loss = total_loss + loss_curr

    # tracer()
    if is_train and not params.pipeline_optimization and total_loss != 0.:
        update_parameters(total_loss)
    if total_loss != 0:
        loss_val = total_loss.item()
    else:
        loss_val = 0.

    return loss_val


def tracer():
    import ipdb
    ipdb.set_trace()
