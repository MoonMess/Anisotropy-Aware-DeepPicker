import torch
import torch.nn.functional as F
from torch import nn
from monai.networks.nets.flexible_unet import SegmentationHead, UNetDecoder, FLEXUNET_BACKBONE


class PatchedUNetDecoder(UNetDecoder):
    
    """add functionality to output all feature maps"""
    
    def forward(self, features: list[torch.Tensor], skip_connect: int = 4):
        skips = features[:-1][::-1]  # Skips from encoder, from deep to shallow
        x = features[-1]  # Start with bottleneck

        out = []
        out += [x]
        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                skip = skips[i]
            else:
                skip = None
            x = block(x, skip)
            out += [x]
        return out


class FlexibleUNet(nn.Module):
    """
    A flexible implementation of UNet-like encoder-decoder architecture. 
    
    (Adjusted to support PatchDecoder and multi segmentation heads for deep supervision)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str,
        pretrained: bool = False,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        spatial_dims: int = 2,
        norm: str | tuple = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: str | tuple = ("leakyrelu", {"inplace": True}),
        dropout: float | tuple = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        pre_conv: str = "default",
        interp_mode: str = "nearest",
        is_pad: bool = True,
    ) -> None:
        super().__init__()

        if backbone not in FLEXUNET_BACKBONE.register_dict:
            raise ValueError(
                f"invalid model_name {backbone} found, must be one of {FLEXUNET_BACKBONE.register_dict.keys()}."
            )

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")

        encoder = FLEXUNET_BACKBONE.register_dict[backbone]
        self.backbone = backbone
        self.spatial_dims = spatial_dims
        encoder_parameters = encoder["parameter"]
        if not (
            ("spatial_dims" in encoder_parameters)
            and ("in_channels" in encoder_parameters)
            and ("pretrained" in encoder_parameters)
        ):
            raise ValueError("The backbone init method must have spatial_dims, in_channels and pretrained parameters.")
        encoder_feature_num = encoder["feature_number"]
        if encoder_feature_num > 5:
            raise ValueError("Flexible unet can only accept no more than 5 encoder feature maps.")

        decoder_channels = decoder_channels[:encoder_feature_num]
        self.skip_connect = encoder_feature_num - 1
        encoder_parameters.update({"spatial_dims": spatial_dims, "in_channels": in_channels, "pretrained": pretrained})
        encoder_channels = tuple([in_channels] + list(encoder["feature_channel"]))
        encoder_type = encoder["type"]
        self.encoder = encoder_type(**encoder_parameters)
        
        
        
        self.decoder = PatchedUNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=pre_conv,
            align_corners=None,
            is_pad=is_pad,
        )
        self.segmentation_heads = nn.ModuleList([SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channel,
            out_channels=out_channels + 1,
            kernel_size=3,
            act=None,
        ) for decoder_channel in decoder_channels])

    def forward(self, inputs: torch.Tensor):

        x = inputs
        enc_out = self.encoder(x)
        # The decoder returns a list of feature maps from each stage (including bottleneck)
        # We take all outputs from the decoder blocks (skipping the initial bottleneck features)
        decoder_out = self.decoder(enc_out, self.skip_connect)[1:]
        
        # Apply segmentation head to each decoder output for deep supervision
        x_seg = [self.segmentation_heads[i](decoder_out[i]) for i in range(len(decoder_out))]

        # Retourne toutes les sorties pour la deep supervision pendant l'entraînement,
        # sinon, ne retourne que la sortie finale pour l'inférence.
        if self.training:
            return x_seg
        return x_seg[-1]