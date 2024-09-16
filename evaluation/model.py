# This file is part of depth-estimation.
#
# Portions of this code are derived from the [Original Repository Name] project,
# which is licensed under the MIT License.
#
# Copyright (c) 2024 Siddhant Biradar.
#
# See the LICENSE.md file in the root of the repository for more details.


import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(
            skip_input, output_features, kernel_size=3, stride=1, padding=1
        )
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(
            output_features, output_features, kernel_size=3, stride=1, padding=1
        )
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode="bilinear",
            align_corners=True,
        )
        return self.leakyreluB(
            self.convB(
                self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))
            )
        )


class Decoder(nn.Module):
    def __init__(self, num_features=2208, decoder_width=0.5):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(
            num_features, features, kernel_size=1, stride=1, padding=1
        )

        self.up1 = UpSample(
            skip_input=features // 1 + 384, output_features=features // 2
        )
        self.up2 = UpSample(
            skip_input=features // 2 + 192, output_features=features // 4
        )
        self.up3 = UpSample(
            skip_input=features // 4 + 96, output_features=features // 8
        )
        self.up4 = UpSample(
            skip_input=features // 8 + 96, output_features=features // 16
        )

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[3],
            features[4],
            features[6],
            features[8],
            features[11],
        )
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models

        self.original_model = models.densenet161(
            weights=models.DenseNet161_Weights.DEFAULT
        )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class TransferLearning(nn.Module):
    def __init__(self):
        super(TransferLearning, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TorchUNET:
    def __init__(
        self, in_channels=3, out_channels=1, init_features=32, pretrained=False
    ):
        self.model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            pretrained=pretrained,
        )
        self.model.cuda()

    def get_model(self):
        return self.model

class CustomUNET1(nn.Module):
    def __init__(self):
        super(CustomUNET1, self).__init__()
        
        # Encoder 1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 2
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 3
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 4
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder 4
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder 3
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder 2
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder 1
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final Convolution
        self.conv = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.encoder4(pool3)
        pool4 = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((up4, enc4), dim=1))
        
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((up3, enc3), dim=1))
        
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((up2, enc2), dim=1))
        
        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((up1, enc1), dim=1))

        # Final output
        return self.conv(dec1)
    
class CustomUNET2(nn.Module):
    def __init__(self):
        super(CustomUNET2, self).__init__()
        
        # Encoder 1 (Increased to 64 filters)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 2 (Adjusted to 128 filters)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 3 (Adjusted to 256 filters)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 4 (Adjusted to 512 filters)
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (Adjusted to 1024 filters)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Decoder 4 (Adjusted for the increased filters)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder 3 (Adjusted for the increased filters)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder 2 (Adjusted for the increased filters)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder 1 (Adjusted for the increased filters)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final Convolution (Output single channel)
        self.conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.encoder4(pool3)
        pool4 = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((up4, enc4), dim=1))
        
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((up3, enc3), dim=1))
        
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((up2, enc2), dim=1))
        
        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((up1, enc1), dim=1))

        # Final output
        return self.conv(dec1)