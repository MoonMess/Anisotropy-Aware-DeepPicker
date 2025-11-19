import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basicblocks import activation, conv3D, get_norm_layer, ResNetBasicBlock3D, ResNetBottleneck3D, conv1x1x1, ResidualBlock, Upscale3D



class ResNetEncoderUNet(nn.Module):
    def __init__(self, input_dim, num_classes=2, encoder_name='resnet101', norm_type='batch', filters=[32, 64, 128, 256], dropout_rate=0, downsample_strides=(2, 2, 2)):
        super(ResNetEncoderUNet, self).__init__()
        self.encoder_name = encoder_name
        self.norm_type = norm_type

        self._build_resnet_unet(input_dim, num_classes)

    def _make_resnet_layer(self, block, planes, blocks, stride=1, norm_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                get_norm_layer(norm_type, planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_type=norm_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=norm_type))
        return nn.Sequential(*layers)

    def _build_resnet_unet(self, input_dim, num_classes):
        if self.encoder_name == 'resnet18':
            block, layers, enc_channels, decoder_filters = ResNetBasicBlock3D, [2, 2, 2, 2], [64, 64, 128, 256, 512], [256, 128, 64, 32]
        elif self.encoder_name == 'resnet34':
            block, layers, enc_channels, decoder_filters = ResNetBasicBlock3D, [3, 4, 6, 3], [64, 64, 128, 256, 512], [256, 128, 64, 32]
        elif self.encoder_name == 'resnet50':
            block, layers, enc_channels, decoder_filters = ResNetBottleneck3D, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], [256, 128, 64, 32]
        elif self.encoder_name == 'resnet101':
            block, layers, enc_channels, decoder_filters = ResNetBottleneck3D, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], [256, 128, 64, 32]
        else:
            raise ValueError(f"Unsupported encoder: '{self.encoder_name}'.")

        self.inplanes = 64

        # Handle 'mix' mode: InstanceNorm for early layers, BatchNorm for deeper ones.
        first_layer_norm = 'instance' if self.norm_type == 'mix' else self.norm_type
        deeper_layer_norm = 'batch' if self.norm_type == 'mix' else self.norm_type

        # --- ResNet Encoder ---
        self.encoder_conv1 = nn.Conv3d(input_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_bn1 = get_norm_layer(first_layer_norm, self.inplanes)
        self.encoder_relu = nn.PReLU()
        self.encoder_maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.encoder_layer1 = self._make_resnet_layer(block, 64, layers[0], norm_type=first_layer_norm)
        self.encoder_layer2 = self._make_resnet_layer(block, 128, layers[1], stride=2, norm_type=deeper_layer_norm)
        self.encoder_layer3 = self._make_resnet_layer(block, 256, layers[2], stride=2, norm_type=deeper_layer_norm)
        self.encoder_layer4 = self._make_resnet_layer(block, 512, layers[3], stride=2, norm_type=deeper_layer_norm)

        # --- U-Net Decoder ---
        self.decoder = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()

        # upsample + conv for each decoder stage
        in_channels = enc_channels[-1]
        for i in range(len(decoder_filters)):
            skip_channels = enc_channels[3-i]
            out_channels = decoder_filters[i]
            
            self.upsample_ops.append(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            self.decoder.append(
                ResidualBlock(out_channels + skip_channels, out_channels, norm_type=deeper_layer_norm)
            )
            in_channels = out_channels

        # Final upsampling to restore original size
        self.final_upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)

        # Prediction heads
        self.seg_head = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        input_tensor = x # Save input tensor for final size check

        if 'resnet' in self.encoder_name:
            # --- ResNet Encoder ---
            skip0 = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(input_tensor)))
            x = self.encoder_maxpool(skip0)
            skip1 = self.encoder_layer1(x)
            skip2 = self.encoder_layer2(skip1)
            skip3 = self.encoder_layer3(skip2)
            bottleneck = self.encoder_layer4(skip3)

            skips = [skip3, skip2, skip1, skip0]

            # --- U-Net Decoder ---
            x = bottleneck
            for i in range(len(self.decoder)):
                x = self.upsample_ops[i](x)
                # Handle size mismatch due to convolutions
                if x.shape[2:] != skips[i].shape[2:]:
                    x = F.interpolate(x, size=skips[i].shape[2:], mode='trilinear', align_corners=False)
                x = torch.cat([x, skips[i]], dim=1)
                x = self.decoder[i](x)

            # Final upsampling to match input size
            x = self.final_upsample(x)
            if x.shape[2:] != input_tensor.shape[2:]:
                 x = F.interpolate(x, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)

        else: # Original custom encoder (This branch is now dead code)
            down_outputs = []
            x = self.act1(self.bn1(self.conv1(x)))
            x = self.dp1(x)

            for enc_layer in self.encoder:
                x = enc_layer(x)
                down_outputs.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)

            for i, (dec_layer, skip_conn) in enumerate(zip(self.decoder, reversed(down_outputs))):
                x = self.upsample_ops[i](x)
                if x.shape[2:] != skip_conn.shape[2:]:
                    x = F.interpolate(x, size=skip_conn.shape[2:], mode='trilinear', align_corners=True)
                x = torch.cat([x, skip_conn], dim=1)
                x = dec_layer(x)

        return self.seg_head(x)
