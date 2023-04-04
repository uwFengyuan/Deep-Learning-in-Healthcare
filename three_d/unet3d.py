from collections import OrderedDict
import torch
import torch.nn as nn
from torchsummary import summary


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64, steps = 4):
        
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNet3D, self).__init__()

        self.steps = steps
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block((features * 8) * 2 , features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.float32)  # x:  torch.Size([8, 1, 48, 128, 128])
        #print('x: ', x.shape)
        enc1 = self.encoder1(x)  # enc1:  torch.Size([8, 10, 48, 128, 128])
        #print('enc1: ', enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1)) # enc2:  torch.Size([8, 20, 24, 64, 64])
        #print('enc2: ', enc2.shape)
        if self.steps == 3:
            bottleneck = self.encoder3(self.pool2(enc2)) # enc3:  torch.Size([8, 40, 12, 32, 32])
            #print('bottleneck: ', bottleneck.shape)

            dec2 = self.decoder2(torch.cat((self.upconv2(bottleneck), enc2), dim=1)) # dec2:  torch.Size([8, 20, 24, 64, 64])
            #print('dec2: ', dec2.shape)
        
        if self.steps == 4:
            enc3 = self.encoder3(self.pool2(enc2)) # enc3:  torch.Size([8, 40, 12, 32, 32])
            #print('enc3: ', enc3.shape)

            bottleneck = self.encoder4(self.pool4(enc3)) # enc4: torch.Size([8, 80, 6, 16, 16])
            #print('bottleneck: ', bottleneck.shape)

            dec3 = self.decoder3(torch.cat((self.upconv3(bottleneck), enc3), dim=1)) # dec3:  torch.Size([8, 40, 12, 32, 32])
            #print('dec3: ', dec3.shape)
            dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1)) # dec2:  torch.Size([8, 20, 24, 64, 64])
            #print('dec2: ', dec2.shape)

        if self.steps == 5:
            enc3 = self.encoder3(self.pool2(enc2)) # enc3:  torch.Size([8, 40, 12, 32, 32])
            #print('enc3: ', enc3.shape)
            enc4 = self.encoder4(self.pool4(enc3)) # enc4: torch.Size([8, 80, 6, 16, 16])
            #print('enc4: ', enc4.shape)
            
            bottleneck = self.bottleneck(self.pool3(enc4)) # bottleneck:  torch.Size([8, 160, 3, 8, 8])
            #print('bottleneck: ', bottleneck.shape)

            dec4 = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1)) # dec4: torch.Size([8, 80, 6, 16, 16])
            #print('dec4: ', dec4.shape)
            dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1)) # dec3:  torch.Size([8, 40, 12, 32, 32])
            #print('dec3: ', dec3.shape)
            dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1)) # dec2:  torch.Size([8, 20, 24, 64, 64])
            #print('dec2: ', dec2.shape)

        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1)) # dec1:  torch.Size([8, 10, 48, 128, 128])
        #print('dec1: ', dec1.shape)
        outputs = self.act(self.conv(dec1)) # outputs:  torch.Size([8, 2, 48, 128, 128])
        #print('outputs: ', outputs.shape)
        return outputs

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


