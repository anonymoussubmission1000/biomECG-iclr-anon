import torch
import torch.nn as nn
from typing import Optional, Collection, List, Union


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
        [2 * nf if concat_pooling else nf, nc]
        if lin_ftrs is None
        else [2 * nf if concat_pooling else nf] + list(lin_ftrs) + [nc]
    )
    ps = ps if isinstance(ps, list) else [ps]
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    acts = [act] * (len(lin_ftrs) - 2) + [None]
    layers: List[nn.Module] = [
        AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2),
        nn.Flatten(start_dim=1),
    ]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, acts):
        layers.append(LinearBlock1d(ni, no, bn=bn, p=p, act=actn))
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        kernel_size=[3, 3],
        dilation=1,
        downsample=None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.conv1 = nn.Conv1d(
            inplanes,
            planes,
            kernel_size=kernel_size[0],
            stride=stride,
            padding=(kernel_size[0] - 1) * dilation // 2,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=kernel_size[1],
            stride=1,
            padding=(kernel_size[1] - 1) * dilation // 2,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
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


class ResNet1d(nn.Sequential):
    """1D adaptation of the torchvision ResNet."""

    def __init__(
        self,
        block,
        layers,
        kernel_size=3,
        dilation_rate=1,
        num_classes=2,
        input_channels=3,
        inplanes=64,
        fix_feature_dim=False,
        kernel_size_stem=None,
        stride_stem=2,
        pooling_stem=True,
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
            )
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
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        for i, l in enumerate(layers):
            current_dilation = dilation_rate**i
            if i == 0:
                layers_tmp.append(
                    self._make_layer(
                        block,
                        planes=inplanes,
                        blocks=layers[0],
                        kernel_size=kernel_size,
                        dilation=current_dilation,
                    )
                )
            else:
                layers_tmp.append(
                    self._make_layer(
                        block,
                        planes=inplanes if fix_feature_dim else 2**i * inplanes,
                        blocks=layers[i],
                        stride=stride,
                        kernel_size=kernel_size,
                        dilation=current_dilation,
                    )
                )
        head = create_head1d(
            (inplanes if fix_feature_dim else 2 ** (len(layers) - 1) * inplanes)
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
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride, kernel_size, dilation, downsample)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, kernel_size=kernel_size, dilation=dilation)
            )
        return nn.Sequential(*layers)

    def get_layer_groups(self):
        return (self[6], self[-1])

    def get_output_layer(self):
        return self[-1][-1]

    def set_output_layer(self, x):
        self[-1][-1] = x


def resnet1d18(**kwargs):
    return ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)


class SiameseResNet1d(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        input_channels=1,
        fix_feature_dim=True,
        kernel_size=3,
        dilation_rate=1,
        dropout=0.1,
    ):
        """
        Dropout is used only in projection head and in the classifier.
        """
        super().__init__()
        backbone = resnet1d18(
            input_channels=input_channels,
            num_classes=1,
            fix_feature_dim=fix_feature_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            pooling_stem=True,
            concat_pooling=True,
            ps_head=0,
        )
        backbone[-1] = backbone[-1][:-1]
        self.encoder = backbone
        if fix_feature_dim:
            encoder_dim = 2 * 64
        else:
            encoder_dim = 2 * 2 ** (4 - 1) * 64
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
            nn.Linear(256, 1),
        )

    def forward(self, x):
        C = x.shape[1] // 2
        x1 = x[:, :C, :]
        x2 = x[:, C:, :]
        z1 = self.embed(self.encoder(x1))
        z2 = self.embed(self.encoder(x2))
        h = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=1)
        return self.classifier(h)
