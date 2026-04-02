from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "hswish":
        return nn.Hardswish(inplace=True)
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation '{name}'.")


def _fill_up_weights(up: nn.ConvTranspose2d) -> None:
    weight = up.weight.data
    factor = math.ceil(weight.size(2) / 2)
    center = (2 * factor - 1 - factor % 2) / (2.0 * factor)
    for i in range(weight.size(2)):
        for j in range(weight.size(3)):
            weight[0, 0, i, j] = (1 - abs(i / factor - center)) * (1 - abs(j / factor - center))
    for channel in range(1, weight.size(0)):
        weight[channel, 0, :, :] = weight[0, 0, :, :]


class _ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            _make_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _SeparableConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, activation: str = "relu") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            _make_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _make_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpsampleTranspose(nn.Module):
    def __init__(self, channels: int, scale: int, *, groups: int, activation: str = "identity") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=scale * 2,
            stride=scale,
            padding=scale // 2,
            output_padding=0,
            groups=groups,
            bias=False,
        )
        _fill_up_weights(self.up)
        if activation == "identity":
            self.post = nn.Identity()
        else:
            self.post = nn.Sequential(
                nn.BatchNorm2d(channels),
                _make_activation(activation),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.up(x))


class _DecoderConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, activation: str = "relu") -> None:
        super().__init__()
        self.block = _ConvBNAct(in_channels, out_channels, kernel_size=3, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _FastWeightedFusion(nn.Module):
    def __init__(self, num_inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}.")
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        output = torch.zeros_like(inputs[0])
        for w, x in zip(weights, inputs):
            output = output + w * x
        return output


class _BiFPNFuseBlock(nn.Module):
    def __init__(self, channels: int, num_inputs: int, *, activation: str = "hswish") -> None:
        super().__init__()
        self.fusion = _FastWeightedFusion(num_inputs)
        self.refine = _SeparableConvBlock(channels, channels, activation=activation)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.refine(self.fusion(*inputs))


class ResNetCenterNetDecoder(nn.Module):
    """ResNet decoder using 3-stage conv + deconv upsampling."""

    out_channels: int = 64

    def __init__(self, in_channels_list: Sequence[int], _decoder_channels: int | None = None) -> None:
        super().__init__()
        del _decoder_channels
        deconv_channels = [256, 128, 64]
        layers: List[nn.Module] = []
        prev_channels = int(in_channels_list[-1])
        for out_channels in deconv_channels:
            layers.append(
                nn.Sequential(
                    _DecoderConvBlock(prev_channels, out_channels, activation="relu"),
                    _UpsampleTranspose(out_channels, scale=2, groups=1, activation="relu"),
                )
            )
            prev_channels = out_channels
        self.decoder = nn.Sequential(*layers)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.decoder(features[-1])


class _IDAUp(nn.Module):
    def __init__(self, out_channels: int, channels: Sequence[int], up_factors: Sequence[int]) -> None:
        super().__init__()
        for index in range(1, len(channels)):
            in_channels = int(channels[index])
            scale = int(up_factors[index])
            setattr(self, f"proj_{index}", _DecoderConvBlock(in_channels, out_channels, activation="relu"))
            setattr(self, f"up_{index}", _UpsampleTranspose(out_channels, scale=scale, groups=out_channels))
            setattr(self, f"node_{index}", _DecoderConvBlock(out_channels, out_channels, activation="relu"))

    def forward(self, layers: List[torch.Tensor], startp: int, endp: int) -> None:
        for index in range(startp + 1, endp):
            block_index = index - startp
            project = getattr(self, f"proj_{block_index}")
            upsample = getattr(self, f"up_{block_index}")
            node = getattr(self, f"node_{block_index}")
            layers[index] = upsample(project(layers[index]))
            layers[index] = node(layers[index] + layers[index - 1])


class _DLAUp(nn.Module):
    def __init__(self, channels: Sequence[int], scales: Sequence[int]) -> None:
        super().__init__()
        channels_list = [int(ch) for ch in channels]
        scales_list = [int(scale) for scale in scales]
        in_channels = channels_list[:]

        for stage in range(len(channels_list) - 1):
            level = -stage - 2
            factors = [scale // scales_list[level] for scale in scales_list[level:]]
            setattr(self, f"ida_{stage}", _IDAUp(channels_list[level], in_channels[level:], factors))
            scales_list[level + 1:] = [scales_list[level] for _ in scales_list[level + 1:]]
            in_channels[level + 1:] = [channels_list[level] for _ in in_channels[level + 1:]]

    def forward(self, layers: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        outputs = [feature.clone() for feature in layers]
        out = [outputs[-1]]
        for stage in range(len(outputs) - 1):
            ida = getattr(self, f"ida_{stage}")
            ida(outputs, len(outputs) - stage - 2, len(outputs))
            out.insert(0, outputs[-1])
        return out


class DLACenterNetDecoder(nn.Module):
    """DLAUp + IDAUp decoder aligned with the official CenterNet DLA path."""

    out_channels: int = 64

    def __init__(self, in_channels_list: Sequence[int], _decoder_channels: int | None = None) -> None:
        super().__init__()
        del _decoder_channels
        channels = [int(ch) for ch in in_channels_list]
        self.dla_up = _DLAUp(channels, scales=[1, 2, 4, 8])
        self.ida_up = _IDAUp(channels[0], channels[:3], up_factors=[1, 2, 4])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        fused = self.dla_up(features)
        outputs = [feature.clone() for feature in fused[:-1]]
        self.ida_up(outputs, 0, len(outputs))
        return outputs[-1]


class MobileNetV3BiFPNDecoder(nn.Module):
    """Lightweight MobileNetV3 decoder with BiFPN-style bidirectional fusion."""

    def __init__(self, in_channels_list: Sequence[int], out_channels: int) -> None:
        super().__init__()
        activation = "hswish"
        self.out_channels = out_channels
        self.proj = nn.ModuleList([
            _ConvBNAct(int(in_channels), out_channels, kernel_size=1, padding=0, activation=activation)
            for in_channels in in_channels_list
        ])
        self.context = _SeparableConvBlock(out_channels, out_channels, activation=activation)

        self.up_5_to_4 = _UpsampleTranspose(out_channels, scale=2, groups=out_channels)
        self.up_4_to_3 = _UpsampleTranspose(out_channels, scale=2, groups=out_channels)
        self.up_3_to_2 = _UpsampleTranspose(out_channels, scale=2, groups=out_channels)

        self.down_2_to_3 = _SeparableConvBlock(out_channels, out_channels, stride=2, activation=activation)
        self.down_3_to_4 = _SeparableConvBlock(out_channels, out_channels, stride=2, activation=activation)
        self.down_4_to_5 = _SeparableConvBlock(out_channels, out_channels, stride=2, activation=activation)

        self.p4_td = _BiFPNFuseBlock(out_channels, num_inputs=2, activation=activation)
        self.p3_td = _BiFPNFuseBlock(out_channels, num_inputs=2, activation=activation)
        self.p2_td = _BiFPNFuseBlock(out_channels, num_inputs=2, activation=activation)
        self.p3_out = _BiFPNFuseBlock(out_channels, num_inputs=3, activation=activation)
        self.p4_out = _BiFPNFuseBlock(out_channels, num_inputs=3, activation=activation)
        self.p5_out = _BiFPNFuseBlock(out_channels, num_inputs=3, activation=activation)

        self.out_up_p3 = _UpsampleTranspose(out_channels, scale=2, groups=out_channels, activation=activation)
        self.out_up_p4 = nn.Sequential(
            _UpsampleTranspose(out_channels, scale=2, groups=out_channels, activation=activation),
            _UpsampleTranspose(out_channels, scale=2, groups=out_channels, activation=activation),
        )
        self.out_up_p5 = nn.Sequential(
            _UpsampleTranspose(out_channels, scale=2, groups=out_channels, activation=activation),
            _UpsampleTranspose(out_channels, scale=2, groups=out_channels, activation=activation),
            _UpsampleTranspose(out_channels, scale=2, groups=out_channels, activation=activation),
        )
        self.output = nn.Sequential(
            _ConvBNAct(out_channels * 4, out_channels, kernel_size=1, padding=0, activation=activation),
            _SeparableConvBlock(out_channels, out_channels, activation=activation),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        p2, p3, p4, p5 = [proj(feature) for proj, feature in zip(self.proj, features)]
        p5_td = self.context(p5)
        p4_td = self.p4_td(p4, self.up_5_to_4(p5_td))
        p3_td = self.p3_td(p3, self.up_4_to_3(p4_td))
        p2_td = self.p2_td(p2, self.up_3_to_2(p3_td))

        p3_out = self.p3_out(p3, p3_td, self.down_2_to_3(p2_td))
        p4_out = self.p4_out(p4, p4_td, self.down_3_to_4(p3_out))
        p5_out = self.p5_out(p5, p5_td, self.down_4_to_5(p4_out))

        fused = torch.cat([
            p2_td,
            self.out_up_p3(p3_out),
            self.out_up_p4(p4_out),
            self.out_up_p5(p5_out),
        ], dim=1)
        return self.output(fused)


def build_decoder(backbone_name: str, in_channels_list: Sequence[int], decoder_channels: int) -> nn.Module:
    if backbone_name == "resnet18":
        return ResNetCenterNetDecoder(in_channels_list, decoder_channels)
    if backbone_name == "dla34":
        return DLACenterNetDecoder(in_channels_list, decoder_channels)
    if backbone_name == "mobilenetv3_large":
        return MobileNetV3BiFPNDecoder(in_channels_list, decoder_channels)
    raise ValueError(f"Unsupported decoder backbone '{backbone_name}'.")
