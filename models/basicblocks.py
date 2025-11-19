import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from torch.utils.checkpoint import checkpoint # Added for activation checkpointing
import warnings # Added import

# --- Activation and Pooling (unchanged) ---

def activation(nonlin: str = 'prelu') -> nn.Module:
    """Returns the appropriate activation layer."""
    if nonlin == 'relu':
        return nn.ReLU(inplace=False)
    elif nonlin == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=False)
    elif nonlin == 'prelu':
        return nn.PReLU()
    else:
        raise NotImplementedError(f"Nonlinearity '{nonlin}' not implemented.")

# --- Normalization Layer ---

def get_norm_layer(norm_type: Optional[str], num_features: int) -> nn.Module:
    """
    Creates a 3D normalization layer.
    """    
    if norm_type is None:
        return nn.Identity()
    elif norm_type == 'batch':
        return nn.BatchNorm3d(num_features)
    elif norm_type == 'instance':
        return nn.InstanceNorm3d(num_features)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

# --- Convolutions ---

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.0, norm_type=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3D(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = get_norm_layer(norm_type=norm_type, num_features=out_channels) 
        self.dropout1 = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = conv3D(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = get_norm_layer(norm_type=norm_type, num_features=out_channels) 
        self.dropout2 = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.act1= activation()
        self.act2 = activation()
        # Skip connection: Adapt dimensions if needed
        self.shortcut = conv3D(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()
        self.residual_connection = ResidualConnection()

        # Erroneous mouth layer removed from ResidualBlock (Note: This comment was present in the original code)

    def forward(self, x):
        shortcut_processed = self.shortcut(x)

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        # No activation here before residual connection

        # Residual connection
        # The ResidualConnection in basicblocks expects (x, residual) where x is skip, residual is main path
        # Here, shortcut_processed is the skip, and out is the main path output
        out = self.residual_connection(shortcut_processed, out) 

        # Final activation
        out = self.act2(out)
        return out


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return conv3D(in_planes, out_planes, kernel_size=1, stride=stride)

class ResNetBasicBlock3D(nn.Module):
    """Basic block for ResNet-18/34, adapted to 3D."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type=None):
        super(ResNetBasicBlock3D, self).__init__()
        
        self.conv1 = conv3D(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = get_norm_layer(norm_type, planes)
        self.relu1 = nn.PReLU()
        self.conv2 = conv3D(planes, planes, kernel_size=3, padding=1)
        self.bn2 = get_norm_layer(norm_type, planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.PReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

class ResNetBottleneck3D(nn.Module):
    """Bottleneck block for ResNet-50/101/152, adapted to 3D."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type=None):
        super(ResNetBottleneck3D, self).__init__()
        
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = get_norm_layer(norm_type, planes)
        self.conv2 = conv3D(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = get_norm_layer(norm_type, planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = get_norm_layer(norm_type, planes * self.expansion)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

def conv3D(in_channels: int, out_channels: int,
           kernel_size: Union[int, Tuple[int, int, int]],
           stride: Union[int, Tuple[int, int, int]] = 1,
           padding: Union[int, Tuple[int, int, int], str] = 0,
           dilation: Union[int, Tuple[int, int, int]] = 1,
           groups: int = 1, bias: bool = False,
           padding_mode: str = 'zeros') -> nn.Module:
    """Creates a standard 3D convolution layer without bias."""
    # Standard PyTorch convolution with bias disabled by default
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups,
                     bias=bias, padding_mode=padding_mode)

# --- Residual Connection ---

class ResidualConnection(nn.Module):
    """
    Residual connection.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # x is the skip connection
        # residual is the main path output
        return residual + x


# --- Upsampling (Upscale3D) ---

class Upscale3D(nn.Module):
    """
    3D upsampling layer, adapted to different modes, following 2D logic.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 scale_factor: Union[int, Tuple[int, int, int]] = 2, # Upsampling factor
                 bias: bool = False,
                 dropout: Optional[float] = None, # Not used in the 2D code
                 norm_type: Optional[str] = None): # Not used in the 2D code
        super().__init__()
        self.norm_type = norm_type # Kept for consistency, even if not used

        # nn.ConvTranspose3d can handle scale_factor as int or tuple for kernel_size/stride
        kernel_stride_param = scale_factor
        if isinstance(kernel_stride_param, int) and kernel_stride_param <= 0:
            raise ValueError("Integer scale_factor must be positive for ConvTranspose3d.")
        elif isinstance(kernel_stride_param, tuple) and any(sf <= 0 for sf in kernel_stride_param):
            raise ValueError("All elements in tuple scale_factor must be positive for ConvTranspose3d.")

        self.upsample = nn.ConvTranspose3d(in_channels, out_channels,
                                           kernel_size=kernel_stride_param,
                                           stride=kernel_stride_param,
                                           padding=0,
                                           bias=bias)

        # Activation after upsampling
        self.activation = activation()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward pass implementation for Upscale3D."""
        out = self.upsample(x)
        # Apply Norm/Dropout here if needed
        out = self.activation(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normal execution
        return self._forward_impl(x)

# --- NEW BASIC BLOCKS (INSPIRED BY MONAI) ---

def get_act(act):
    if isinstance(act, str):
        act = act.lower()
        if act == "relu":
            return nn.ReLU(inplace=True)
        elif act == "prelu":
            return nn.PReLU()
        elif act == "leakyrelu":
            return nn.LeakyReLU(inplace=True)
        elif act == "elu":
            return nn.ELU(inplace=True)
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {act}")
    elif isinstance(act, nn.Module):
        return act
    else:
        raise ValueError("Activation must be string or nn.Module")

def get_norm(norm, num_features, spatial_dims):
    if isinstance(norm, str):
        norm = norm.lower()
        if norm == "batch":
            if spatial_dims == 1:
                return nn.BatchNorm1d(num_features)
            elif spatial_dims == 2:
                return nn.BatchNorm2d(num_features)
            elif spatial_dims == 3:
                return nn.BatchNorm3d(num_features)
        elif norm == "instance":
            if spatial_dims == 1:
                return nn.InstanceNorm1d(num_features)
            elif spatial_dims == 2:
                return nn.InstanceNorm2d(num_features)
            elif spatial_dims == 3:
                return nn.InstanceNorm3d(num_features)
        elif norm == "layer":
            return nn.LayerNorm(num_features)
        else:
            raise ValueError(f"Unsupported normalization: {norm}")
    elif isinstance(norm, nn.Module):
        return norm
    else:
        raise ValueError("Normalization must be string or nn.Module")

class Convolution(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        strides=1,
        kernel_size=3,
        act="prelu",
        norm="instance",
        dropout=0.0,
        bias=True,
        conv_only=False,
        is_transposed=False,
        adn_ordering="NDA",  # N: norm, D: dropout, A: activation
        output_padding=0,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial_dims
        if isinstance(strides, int):
            strides = [strides] * spatial_dims

        conv_cls = {
            (1, False): nn.Conv1d,
            (2, False): nn.Conv2d,
            (3, False): nn.Conv3d,
            (1, True): nn.ConvTranspose1d,
            (2, True): nn.ConvTranspose2d,
            (3, True): nn.ConvTranspose3d,
        }[(spatial_dims, is_transposed)]

        padding = tuple(k // 2 for k in kernel_size)

        conv_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": strides,
            "padding": padding,
            "bias": bias,
        }
        if is_transposed:
            conv_kwargs["output_padding"] = output_padding

        self.conv = conv_cls(**conv_kwargs)

        self.norm = get_norm(norm, out_channels, spatial_dims) if norm and not conv_only else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.act = get_act(act) if act and not conv_only else None
        self.adn_ordering = adn_ordering.upper()

    def forward(self, x):
        x = self.conv(x)
        for c in self.adn_ordering:
            if c == "N" and self.norm is not None:
                x = self.norm(x)
            if c == "D" and self.drop is not None:
                x = self.drop(x)
            if c == "A" and self.act is not None:
                x = self.act(x)
        return x

class ResidualUnit(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        strides=1,
        kernel_size=3,
        subunits=2,
        act="prelu",
        norm="instance",
        dropout=0.0,
        bias=True,
        last_conv_only=False,
        adn_ordering="NDA",
    ):
        super().__init__()
        self.need_proj = in_channels != out_channels or (isinstance(strides, int) and strides != 1) or (isinstance(strides, tuple) and any(s != 1 for s in strides))
        self.proj = (
            Convolution(
                spatial_dims,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=1,
                act=None,
                norm=None,
                dropout=0.0,
                bias=bias,
                conv_only=True,
            )
            if self.need_proj
            else nn.Identity()
        )
        blocks = []
        for i in range(subunits):
            blocks.append(
                Convolution(
                    spatial_dims,
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    strides=strides if i == 0 else 1,
                    kernel_size=kernel_size,
                    act=act if not (last_conv_only and i == subunits - 1) else None,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    conv_only=last_conv_only and i == subunits - 1,
                    adn_ordering=adn_ordering,
                )
            )
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        identity = self.proj(x)
        out = self.block(x)
        return out + identity

class SkipConnection(nn.Module):
    """
    Apply a submodule and concatenate its output with the input (skip connection).
    """
    def __init__(self, submodule, dim=1):
        super().__init__()
        self.submodule = submodule
        self.dim = dim

    def forward(self, x):
        return torch.cat([x, self.submodule(x)], dim=self.dim)