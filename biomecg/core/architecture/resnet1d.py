import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Collection, List, Union

# from fastai.layers import Flatten
# from models.basic_conv1d import create_head1d
###############################################################################################

### TOOLS FROM basic_conv1d file 


# Standard resnet

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    """
    1D convolution with automatic same padding

    Args:
        in_planes: number of input channels
        out_planes: number of output channels
        stride: stride of the convolution
        kernel_size: size of the convolution kernel
    """
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False,
    )

class AdaptiveConcatPool1d(nn.Module):
    """
    Concatenates AdaptiveAvgPool1d and AdaptiveMaxPool1d along the channel dimension.

    Args:
        sz: output size after pooling (default = 1)
    """
    def __init__(self, sz: Optional[int] = 1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mp(x), self.ap(x)], dim=1)

class LinearBlock1d(nn.Module):
    """
    A linear block consisting of optional BatchNorm1d, Dropout, Linear, and Activation layers.

    Args:
        ni: number of input features
        no: number of output features
        bn: whether to include BatchNorm1d
        p: dropout probability
        act: activation function ('relu', 'elu', or None)
    """
    def __init__(
        self,
        ni: int,
        no: int,
        bn: bool = True,
        p: float = 0.0,
        act: Optional[str] = "relu",
        ):
        super().__init__()
        layers: List[nn.Module] = []
        if bn:
            layers.append(nn.BatchNorm1d(ni))
        if p > 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(ni, no))
        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "elu":
            layers.append(nn.ELU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

def create_head1d(
    nf: int,
    nc: int,
    lin_ftrs: Optional[Collection[int]] = None,
    ps: Union[float, Collection[float]] = 0.5,
    bn_final: bool = False,
    bn: bool = True,
    act: str = "relu",
    concat_pooling: bool = True,
    ) -> nn.Sequential:
    """
    Creates a fully connected head for a 1D model (e.g. for time series classification/regression).

    Args:
        nf: number of input features from the backbone
        nc: number of output classes or regression targets
        lin_ftrs: list of linear layer sizes (excluding input and output)
        ps: dropout probabilities (float or list)
        bn_final: whether to add a final BatchNorm1d layer
        bn: whether to include BatchNorm1d in intermediate blocks
        act: activation function ('relu' or 'elu')
        concat_pooling: whether to use AdaptiveConcatPool1d or simple MaxPool1d
    """
    lin_ftrs = (
        [2 * nf if concat_pooling else nf, nc] #2nf, nc -> will give a (2nf, nc) linear filter
        if lin_ftrs is None # so here with adapt concat pooling, [2x64, nc] (due to adaptive concat pooling), in case fixed feature dim
        else [2 * nf if concat_pooling else nf] + list(lin_ftrs) + [nc]
    )

    ps = ps if isinstance(ps, list) else [ps] # list or scalar given
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps

    acts = [act] * (len(lin_ftrs) - 2) + [None] # no activation on final layer

    layers: List[nn.Module] = [
        AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2),
        nn.Flatten(start_dim=1),
    ]
    # stack linear blocks from 1 to n-1, adapt input/output sizes
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, acts): 
        layers.append(LinearBlock1d(ni, no, bn=bn, p=p, act=actn))

    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01)) #final layer and bn with low momentum

    return nn.Sequential(*layers)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=[3, 3], dilation=1, downsample=None):
        super().__init__()

        if isinstance(kernel_size, int):
            # kernel_size = [kernel_size, kernel_size // 2 + 1] # second conv has smaller kernel
            #not implemented in resnet
            kernel_size = [kernel_size, kernel_size]
        # self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size[0], 
                               stride=stride, padding=(kernel_size[0]-1)*dilation//2, 
                               dilation=dilation, bias=False)

        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv(planes, planes, kernel_size=kernel_size[1])
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size[1], 
                        stride=1, padding=(kernel_size[1]-1)*dilation//2, 
                        dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample # to change residual connexion dimension if needed
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

###############################################################################################

### RESNET Class 


class ResNet1d(nn.Sequential):
    """1D adaptation of the torchvision ResNet."""

    def __init__(
        self,
        block,
        layers, #[2,2,2,2]
        kernel_size=3,
        dilation_rate=1,
        num_classes=2,
        input_channels=3, #image ?
        inplanes=64, #resnet18 paper
        fix_feature_dim=False,
        #original resnet : False. Apparently, for simple times series fixed_channels works as good
        kernel_size_stem=None,
        stride_stem=2,
        pooling_stem=True, #architecture resnet18 : True 512 -> 128
        stride=2,
        lin_ftrs_head=None,
        ps_head=0.5,
        bn_final_head=False,
        bn_head=True,
        act_head="relu",
        concat_pooling=True,
        ):
        self.inplanes = inplanes
        layers_tmp = []

        if kernel_size_stem is None:
            kernel_size_stem = (
                kernel_size[0] if isinstance(kernel_size, list) else kernel_size
            ) # 3 for resnet

        # Stem
        layers_tmp.append(
            nn.Conv1d(
                input_channels,
                inplanes,
                kernel_size=kernel_size_stem,
                stride=stride_stem,
                padding=(kernel_size_stem - 1) // 2,
                bias=False,
            )
        )
        layers_tmp.append(nn.BatchNorm1d(inplanes))
        layers_tmp.append(nn.ReLU(inplace=True))
        if pooling_stem:
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1)) #hardcoded

        # Backbone
        for i, l in enumerate(layers):
            current_dilation = dilation_rate**i 
            if i == 0:
                layers_tmp.append(
                    self._make_layer(
                        block,
                        planes=inplanes,
                        blocks=layers[0],
                        kernel_size=kernel_size,
                        dilation=current_dilation
                    )
                )
            else:
                layers_tmp.append(
                    self._make_layer(
                        block,
                        planes=inplanes if fix_feature_dim else (2 ** i) * inplanes,
                        blocks=layers[i],
                        stride=stride,
                        kernel_size=kernel_size,
                        dilation=current_dilation
                    )
                )

        # Head
        head = create_head1d(
            (inplanes if fix_feature_dim else (2 ** (len(layers) - 1)) * inplanes)
            * block.expansion,
            nc=num_classes,
            lin_ftrs=lin_ftrs_head,
            ps=ps_head,
            bn_final=bn_final_head,
            bn=bn_head,
            act=act_head,
            concat_pooling=concat_pooling,
        )
        layers_tmp.append(head)
        super().__init__(*layers_tmp)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  #downsample avec conv1d
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )
        #first layer : stride=2 with downsample -
        #exexcluded_source if it's the very first block of the architecture, then no stride/downsample
        layers = [block(self.inplanes, planes, stride, kernel_size, dilation, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size, dilation=dilation))##???????
            # kernel_size identical (modified) / stride=1 by default
        return nn.Sequential(*layers)

    def get_layer_groups(self):
        return self[6], self[-1]

    def get_output_layer(self):
        return self[-1][-1]

    def set_output_layer(self, x):
        self[-1][-1] = x


def resnet1d18(**kwargs):
    return ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
# fix_feature_dim=True : is an adaptation of the original structure to simple 1D signal.
# not much ref found... try both



### ---------------- SIAMESE ARCHITECTURE ------------------ ###
class SiameseResNet1d(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        input_channels=1,
        fix_feature_dim=True,
        kernel_size=3,
        dilation_rate=1,
        dropout=0.1
    ):
        """
        Dropout is used only in projection head and in the classifier.
        """
        super().__init__()

        backbone = resnet1d18(
            input_channels=input_channels,
            num_classes=1,          # dummy
            fix_feature_dim=fix_feature_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            pooling_stem=True,
            concat_pooling=True,
            ps_head = 0  # no dropout
        )
        # Projection head complet : AdaptiveConcatPool1d() + Flatten() + Batch Norm + Pas de Dropout + Linear(2*nf → nc)
        # sous certaines conditions (fix_feature_dim = True ici)
        # backbone[-1] is the projection head
        backbone[-1] = backbone[-1][:-1]  # remove final Linear
        self.encoder = backbone
        # so has just the 
        # AdaptiveConcatPool1d
        # Flatten
        # Linear(128 → 1) #removed

        if fix_feature_dim:
            encoder_dim = 2 * 64 #hardcoded!
        else:
            encoder_dim = 2 * (2 ** (4 - 1)) * 64 #hardcoded!

        self.embed = nn.Sequential(
            nn.Linear(encoder_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(embedding_dim * 4, 64),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(64),
        #     nn.Dropout(dropout),
        #     nn.Linear(64, 1)
        # )

    def forward(self, x):
        C = x.shape[1] // 2
        x1 = x[:, :C, :]
        x2 = x[:, C:, :]

        z1 = self.embed(self.encoder(x1))
        z2 = self.embed(self.encoder(x2))

        h = torch.cat(
            [z1, z2, torch.abs(z1 - z2), z1 * z2],
            dim=1,
        )
        return self.classifier(h)



# def resnet1d34(**kwargs):
#     return ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)

# def resnet1d50(**kwargs):
#     return ResNet1d(Bottleneck1d, [3, 4, 6, 3], **kwargs)


# def resnet1d101(**kwargs):
#     return ResNet1d(Bottleneck1d, [3, 4, 23, 3], **kwargs)


# def resnet1d152(**kwargs):
#     return ResNet1d(Bottleneck1d, [3, 8, 36, 3], **kwargs)


# def resnet1d_wang(**kwargs):
#     if "kernel_size" not in kwargs:
#         kwargs["kernel_size"] = [5, 3]
#     if "kernel_size_stem" not in kwargs:
#         kwargs["kernel_size_stem"] = 7
#     if "stride_stem" not in kwargs:
#         kwargs["stride_stem"] = 1
#     if "pooling_stem" not in kwargs:
#         kwargs["pooling_stem"] = False
#     if "inplanes" not in kwargs:
#         kwargs["inplanes"] = 128

#     return ResNet1d(BasicBlock1d, [1, 1, 1], **kwargs)


# def resnet1d(**kwargs):
#     return ResNet1d(BasicBlock1d, **kwargs)



###############################################################################################
# wide resnet adopted from fastai wrn

# def noop(x): return x

# def conv1d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False) -> nn.Conv1d:
#     "Create `nn.Conv1d` layer: `ni` inputs, `nf` outputs, `ks` kernel size. `padding` defaults to `k//2`."
#     if padding is None: padding = ks//2
#     return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

# def _bn1d(ni, init_zero=False):
#     "Batchnorm layer with 0 initialization"
#     m = nn.BatchNorm1d(ni)
#     m.weight.data.fill_(0 if init_zero else 1)
#     m.bias.data.zero_()
#     return m

# def bn_relu_conv1d(ni, nf, ks, stride, init_zero=False):
#     bn_initzero = _bn1d(ni, init_zero=init_zero)
#     return nn.Sequential(bn_initzero, nn.ReLU(inplace=True), conv1d(ni, nf, ks, stride))

# class BasicBlock1dwrn(nn.Module):
#     def __init__(self, ni, nf, stride, drop_p=0.0, ks=3):
#         super().__init__()
#         if(isinstance(ks,int)):
#             ks = [ks,ks//2+1]
#         self.bn = nn.BatchNorm1d(ni)
#         self.conv1 = conv1d(ni, nf, ks[0], stride)
#         self.conv2 = bn_relu_conv1d(nf, nf, ks[0], 1)
#         self.drop = nn.Dropout(drop_p, inplace=True) if drop_p else None
#         self.shortcut = conv1d(ni, nf, ks[1], stride) if (ni != nf or stride>1) else noop #adapted to make it work for fix_feature_dim=True

#     def forward(self, x):
#         x2 = F.relu(self.bn(x), inplace=True)
#         r = self.shortcut(x2)
#         x = self.conv1(x2)
#         if self.drop: x = self.drop(x)
#         x = self.conv2(x) * 0.2
#         return x.add_(r)

# def _make_group(N, ni, nf, block, stride, drop_p,ks=3):
#     return [block(ni if i == 0 else nf, nf, stride if i == 0 else 1, drop_p,ks=ks) for i in range(N)]

# class WideResNet1d(nn.Sequential):
#     def __init__(self, input_channels:int, num_groups:int, N:int, num_classes:int, k:int=1, drop_p:float=0.0, start_nf:int=16,fix_feature_dim=True,kernel_size=5,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
#         super().__init__()
#         n_channels = [start_nf]
        
#         for i in range(num_groups): n_channels.append(start_nf if fix_feature_dim else start_nf*(2**i)*k)

#         layers = [conv1d(input_channels, n_channels[0], 3, 1)]  # conv1 stem
#         for i in range(num_groups):
#             layers += _make_group(N, n_channels[i], n_channels[i+1], BasicBlock1dwrn, (1 if i==0 else 2), drop_p,ks=kernel_size)

#         #layers += [nn.BatchNorm1d(n_channels[-1]), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1),
#         #           Flatten(), nn.Linear(n_channels[-1], num_classes)]
#         head = create_head1d(n_channels[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
#         layers.append(head)
        
#         super().__init__(*layers)
    
#     def get_layer_groups(self):
#         return (self[6],self[-1])
    
#     def get_output_layer(self):
#         return self[-1][-1]
    
#     def set_output_layer(self,x):
#         self[-1][-1] = x


# def wrn1d_22(**kwargs): return WideResNet1d(num_groups=3, N=3, k=6, drop_p=0.,**kwargs)



########### TEST SECTION ##################

###############################################################################################
### TESTS FOR basic_conv1d & ResNet1d
###############################################################################################


def test_conv():
    x = torch.randn(4, 8, 100)  # (batch, channels, length)
    layer = conv(8, 16, kernel_size=3, stride=1)
    y = layer(x)
    assert y.shape == (4, 16, 100), f"Unexpected conv output shape: {y.shape}"
    print("✅ conv() output OK:", y.shape)


def test_adaptiveconcatpool1d():
    x = torch.randn(4, 8, 50)
    pool = AdaptiveConcatPool1d(1)
    y = pool(x)
    assert y.shape == (4, 16, 1), f"Unexpected pooling shape: {y.shape}"
    print("✅ AdaptiveConcatPool1d OK:", y.shape)


def test_linearblock1d():
    x = torch.randn(4, 32)
    block = LinearBlock1d(32, 10, bn=True, p=0.1, act="relu")
    y = block(x)
    assert y.shape == (4, 10), f"Unexpected LinearBlock1d output shape: {y.shape}"
    print("✅ LinearBlock1d OK:", y.shape)


def test_create_head1d():
    x = torch.randn(4, 64, 200)
    head = create_head1d(nf=64, nc=5, lin_ftrs=[128], ps=0.3)
    y = head(x)
    assert y.shape == (4, 5), f"Unexpected head output shape: {y.shape}"
    print("✅ create_head1d() OK:", y.shape)


def test_basicblock1d_identity():
    x = torch.randn(2, 16, 50)
    block = BasicBlock1d(16, 16, stride=1)
    y = block(x)
    assert y.shape == x.shape, f"Unexpected BasicBlock1d output shape: {y.shape}"
    print("✅ BasicBlock1d (identity) OK:", y.shape)


def test_basicblock1d_downsample():
    x = torch.randn(2, 16, 50)
    block = BasicBlock1d(16, 32, stride=2,
                         downsample=nn.Sequential(nn.Conv1d(16, 32, 1, 2, bias=False),
                                                  nn.BatchNorm1d(32)))
    y = block(x)
    assert y.shape == (2, 32, 25), f"Unexpected BasicBlock1d downsample shape: {y.shape}"
    print("✅ BasicBlock1d (downsample) OK:", y.shape)


def test_resnet1d18_forward():
    model = resnet1d18(
        input_channels=3,
        num_classes=4,
        fix_feature_dim=True,
        concat_pooling=True,
    )
    x = torch.randn(2, 3, 500)
    y = model(x)
    assert y.shape == (2, 4), f"Unexpected ResNet1d output shape: {y.shape}"
    print("✅ resnet1d18 forward OK:", y.shape)


def test_resnet1d34_forward():
    model = resnet1d34(
        input_channels=3,
        num_classes=3,
        fix_feature_dim=False,
        concat_pooling=False,
    )
    x = torch.randn(2, 3, 512)
    y = model(x)
    assert y.shape == (2, 3), f"Unexpected ResNet1d34 output shape: {y.shape}"
    print("✅ resnet1d34 forward OK:", y.shape)


def main():
    print("\nRunning basic_conv1d / ResNet1d tests...\n")
    test_conv()
    test_adaptiveconcatpool1d()
    test_linearblock1d()
    test_create_head1d()
    test_basicblock1d_identity()
    test_basicblock1d_downsample()
    test_resnet1d18_forward()
    test_resnet1d34_forward()
    print("\n🎯 All tests passed successfully!\n")
