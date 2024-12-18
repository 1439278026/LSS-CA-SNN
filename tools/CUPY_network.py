import numpy as np
import torch
import torch.nn as nn
from .CUPY_neuron import CUPYPLIFNode,CUPYLIFNode,CUPYIFNode,MS_Attention_RepConv_qkv_id
from . import layer, functional, surrogate, neuron
import torch.nn.functional as F
from typing import Optional
import math
from einops.layers.torch import Reduce

class ShallowConvNet(nn.Module):
    def __init__(self, classes_num, in_channels, time_step, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.classes_num = classes_num
        n_ch1 = 40

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))
        self.layer1.eval()
        out = self.layer1(torch.zeros(1, 1, in_channels, time_step))
        out = torch.nn.functional.avg_pool2d(out, (1, 75), 15)
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]
        self.clf = nn.Linear(self.n_outputs, self.classes_num)

    def forward(self, x):
        x = x.unsqueeze(1)  # [N, C, T] -> [N, 1, C, T]
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 75), 15)
        x = torch.log(x)
        x = torch.nn.functional.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.clf(x)
        return x
def CalculateOutSize(model, Chans, Samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(1, 1, Chans, Samples).to(device)
    out = model(x)
    out = out.view(out.size(0), -1)
    return out.shape[-1]
# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, Chans, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()
        self.Chans = Chans

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (self.Chans, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        x = x.permute(0,2,3,1)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask = None):
        queries = self.queries(x)
        queries = queries.permute(0, 2, 1, 3)
        values = self.values(x).permute(0, 2, 1, 3)
        keys = self.keys(x).permute(0, 2, 1, 3)
        # queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        # keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        # values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = out.permute(0,2,1,3)
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
class GELU(nn.Module):
    def forward(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
class ClassificationHead(nn.Sequential):
    def __init__(self, Chans, Samples, depth, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.fc_in = CalculateOutSize(model=nn.Sequential(
            PatchEmbedding(Chans, emb_size),
            TransformerEncoder(depth, emb_size)),
            Chans=Chans,
            Samples=Samples)
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out

class Conformer(nn.Module):
    def __init__(self, in_channels, time_step, classes_num, emb_size=40, depth=6):
        super(Conformer, self).__init__()

        self.model = nn.Sequential(
            PatchEmbedding(in_channels, emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(in_channels, time_step, depth, emb_size, classes_num))

    def forward(self, x):
        if x.ndimension() == 3:
            x = x.unsqueeze(1)  # [N, C, T] -> [N, 1, C, T]
        if x.ndimension() == 2:
            x = x.unsqueeze(0)  # [C, T] -> [1, C, T]
            x = x.unsqueeze(0)  # [1, C, T] -> [1, 1, C, T]
        output = self.model(x)
        # output = output.reshape(output.size(0), -1)
        # out = self.clf(output)
        return output
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size:int=50):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.clf = nn.Sequential(self.flatten,self.fc1,self.relu,self.fc2)

    def forward(self, x):
        x = self.clf(x)
        return x
class DeepConvNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 time_step: int,
                 classes_num : int,
                 dropoutRate: Optional[float] = 0.5,
                 d1: Optional[int] = 25,
                 d2: Optional[int] = 50,
                 d3: Optional[int] = 100):
        super(DeepConvNet, self).__init__()

        self.Chans = in_channels
        self.Samples = time_step
        self.dropoutRate = dropoutRate
        self.classes_num = classes_num

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d1, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=(in_channels, 1)),
            nn.BatchNorm2d(num_features=d1), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d2), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=d2, out_channels=d3, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d3), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.convT = CalculateOutSize(nn.Sequential(self.block1,self.block2,self.block3),self.Chans,self.Samples)
        self.clf = Classifier(input_size=self.convT, output_size=self.classes_num)

    def forward(self, x):
        if x.ndimension() == 3:
            x = x.unsqueeze(1)  # [N, C, T] -> [N, 1, C, T]
        if x.ndimension() == 2:
            x = x.unsqueeze(0)  # [C, T] -> [1, C, T]
            x = x.unsqueeze(0)  # [1, C, T] -> [1, 1, C, T]
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        # print(output.shape)
        out = self.clf(output)
        return out

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNet(nn.Module):
    def __init__(self,
                 classes_num: int,
                 in_channels: int,
                 time_step: int,
                 kernLenght: int = 64,
                 F1: int = 8,
                 D: int = 2,
                 F2: int = 16,
                 dropout_size: Optional[float] = 0.5,
                ):
        super(EEGNet, self).__init__()
        self.n_classes = classes_num
        self.Chans = in_channels
        self.Samples = time_step
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropout_size
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x) :
        x = x.unsqueeze(1)  # [N, C, T] -> [N, 1, C, T]
        output = self.block1(x)
        # print(output.shape)
        output = self.block2(output)
        # print(output.shape)
        output1 = output.reshape(output.size(0), -1)
        output = self.classifier_block(output1)
        return output



class ChannelWiseScaling(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWiseScaling, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        return x * self.scale.unsqueeze(0).unsqueeze(-1)

class LSS_CA_SNN(nn.Module):
    def __init__(self, in_channels, out_num, w, surrogate_function, time_step, neuron='PLIF'):
        super(LSS_CA_SNN, self).__init__()
        tau = math.exp(-w) + 1
        v_threshold = 1.0
        v_reset = -0.1
        self.layer5_residual_pool = nn.AvgPool2d((1, 2))  # 对残差路径进行降维

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=(64,), padding=(32,), ),
            nn.BatchNorm1d(in_channels),
            ChannelWiseScaling(in_channels),
            nn.AvgPool1d(2),
            CUPYPLIFNode(init_tau=tau, surrogate_function=surrogate_function, v_threshold=v_threshold, v_reset=v_reset),
                    )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(in_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.AvgPool2d((1, 2)),
            CUPYPLIFNode(init_tau=tau, surrogate_function=surrogate_function, v_threshold=v_threshold, v_reset=v_reset),
        )
        self.layer3 = nn.Sequential(
            nn.ZeroPad2d((32, 32, 0, 0)),
            nn.Conv2d(32, 32, kernel_size=(1, 64), bias=False),  # 增加通道数到32
            nn.BatchNorm2d(32),
            nn.AvgPool2d((1, 2)),
            CUPYPLIFNode(init_tau=tau, surrogate_function=surrogate_function, v_threshold=v_threshold, v_reset=v_reset),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.AvgPool2d((1, 2)),
            CUPYPLIFNode(init_tau=tau, surrogate_function=surrogate_function, v_threshold=v_threshold, v_reset=v_reset),
        )
        self.drop = nn.Dropout(p=0.25)
        self.pool = nn.AvgPool1d(2)
        self.fc = nn.Linear(32, out_num)


    def forward(self, x):  # N, C, T
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = x.unsqueeze(1)
        # x = self.layer2(x)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x4 = self.layer4(x)
        # x5 = self.layer4(x4)
        # print(x4.shape)
        x_residual = self.layer5_residual_pool(x)
        x5 = x4  + x_residual

        x55 = self.layer4(x5)
        x_residual = self.layer5_residual_pool(x5)
        x5 = x55 + x_residual

        x = x5.mean(-1).squeeze(-1)
        # print(x.shape)
        x = self.drop(x)
        x = self.fc(x)
        return x
    def feature(self, x):  # N, C, T
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = x.unsqueeze(1)
        # x = self.layer2(x)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x4 = self.layer4(x)
        # x5 = self.layer4(x4)
        x_residual = self.layer5_residual_pool(x)  # 调整残差路径尺寸
        x5 = x4  + x_residual


        x = x5.mean(-1).squeeze(-1)
        return  x



