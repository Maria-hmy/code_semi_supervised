
import timm 
import torch.nn as nn
from nets import block
import torch

class SwinV2Classifier(nn.Module):
    def __init__(self, model_name='swinv2_cr_tiny_224',
                 pretrained = False,
                 in_chans=1,
                 img_size=(256, 256),
                 patch_size=4,
                 n_classes=6):  # multi-class classification
        super(SwinV2Classifier, self).__init__()
        
        # encoder (backbone)
        self.encoder = timm.create_model(
            model_name,
            pretrained = pretrained,
            in_chans = in_chans,
            num_classes = 0,  # Disable the default classifier head
            global_pool = '',  # No global pooling in the encoder
            img_size = img_size,
            patch_size = patch_size
        )        
        # global pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Adaptive average pooling
        
        # fully-connected classification head
        self.classifier = nn.Linear(self.encoder.num_features, n_classes)
        
    def forward(self, x):
        features = self.encoder(x)  # shape: [B, C, H, W]
        pooled_features = self.global_pool(features)  # shape: [B, C, 1, 1]
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # shape: [B, C]
        
        # Pass through classification head
        output = self.classifier(pooled_features)  # shape: [B, n_classes]
        
        return output


class Doublewith2Up(nn.Module):
    ''' ConvTranspose2d + {Conv2d, BN, ReLU}x2 '''
    
    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan == None:
            mid_chan = in_chan
        self.conv = block.DoubleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.up(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class SwinV2OneDecoder(nn.Module):
    def __init__(self, model_name='swinv2_cr_tiny_224',
                 pretrained=False,
                 in_chans=1,
                 img_size=(256,256),
                 patch_size=4,
                 n_classes=1):
        super(SwinV2OneDecoder, self).__init__()
        encoder = timm.create_model(model_name, pretrained=pretrained,
                                    in_chans=in_chans,
                                    num_classes=0, 
                                    global_pool='',
                                    img_size=img_size,
                                    patch_size=patch_size)
        base_layers=list(encoder.children())
        
        self.n_channels = in_chans
        self.n_classes = n_classes
        
        self.patch_embed=base_layers[0]
        self.skip0=base_layers[1][0]
        self.skip1=base_layers[1][1]
        self.skip2=base_layers[1][2]
        self.center=base_layers[1][3]
                                          
        self.deblock1 = block.QuadripleUp(768,384)
        self.deblock2 = block.QuadripleUp(384,192)
        self.deblock3 = block.DoubleUp(192,96)
        self.deblock4 = Doublewith2Up(96,48)
        self.conv=nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock5 = block.DoubleUp(48,48,96)
        self.outc = block.OutConv(48, n_classes)
    
        
    def forward(self, x):
        x1=self.patch_embed(x)        
        x2=self.skip0(x1)
        x3=self.skip1(x2)
        x4=self.skip2(x3)
        x5=self.center(x4)
        out= self.deblock1(x5, x4)
        out = self.deblock2(out, x3)
        out = self.deblock3(out, x2)
        out = self.deblock4(out, x1)
        out = self.deblock5(out, self.conv(x))
        logits = self.outc(out)
        return logits

class SwinV2TwoDecoder(nn.Module):
    def __init__(self, model_name='swinv2_cr_tiny_224',
                 pretrained=False,
                 img_size=(256,256),
                 in_chans=1,
                 n_classes_1dec=1,
                 n_classes_2dec=1):
        super(SwinV2TwoDecoder, self).__init__()
        encoder = timm.create_model(model_name, pretrained=pretrained,
                                    num_classes=0, 
                                    global_pool='',
                                    img_size=img_size,
                                    in_chans=in_chans)
        base_layers=list(encoder.children())
        
        self.n_channels = in_chans
        self.n_classes = 1 # n_classes_1dec+n_classes_2dec
        
        self.patch_embed=base_layers[0]
        self.skip0=base_layers[1][0]
        self.skip1=base_layers[1][1]
        self.skip2=base_layers[1][2]
        self.center=base_layers[1][3]
        
        # first decoder
        self.deblock1_1 = block.QuadripleUp(768,384)
        self.deblock1_2 = block.QuadripleUp(384,192)
        self.deblock1_3 = block.DoubleUp(192,96)
        self.deblock1_4 = Doublewith2Up(96,48)
        self.conv1=nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock1_5 = block.DoubleUp(48,48,96)
        self.outc1 = block.OutConv(48, n_classes_1dec)
        
        # second decoder
        self.deblock2_1 = block.QuadripleUp(768,384)
        self.deblock2_2 = block.QuadripleUp(384,192)
        self.deblock2_3 = block.DoubleUp(192,96)
        self.deblock2_4 = Doublewith2Up(96,48)
        self.conv2=nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock2_5 = block.DoubleUp(48,48,96)
        self.outc2 = block.OutConv(48, n_classes_2dec)
        
    def forward(self, x):
        x1=self.patch_embed(x)
        x2=self.skip0(x1)
        x3=self.skip1(x2)
        x4=self.skip2(x3)
        x5=self.center(x4)
        
        # first decoder
        out1 = self.deblock1_1(x5, x4)
        out1 = self.deblock1_2(out1, x3)
        out1 = self.deblock1_3(out1, x2)
        out1 = self.deblock1_4(out1, x1)
        out1 = self.deblock1_5(out1, self.conv1(x))
        logits1 = self.outc1(out1)
        
        # second decoder
        out2 = self.deblock2_1(x5, x4)
        out2 = self.deblock2_2(out2, x3)
        out2 = self.deblock2_3(out2, x2)
        out2 = self.deblock2_4(out2, x1)
        out2 = self.deblock2_5(out2, self.conv2(x))
        logits2 = self.outc2(out2)
        
        return logits1, logits2
    
class SwinV2ThreeDecoder(nn.Module):
    def __init__(self, model_name='swinv2_cr_tiny_224',
                 pretrained=False,
                 img_size=(256,256),
                 in_chans=1,
                 n_classes_1dec=1,
                 n_classes_2dec=1,
                 n_classes_3dec=1):
        super(SwinV2ThreeDecoder, self).__init__()
        encoder = timm.create_model(model_name, pretrained=pretrained,
                                    num_classes=0, 
                                    global_pool='',
                                    img_size=img_size,
                                    in_chans=in_chans)
        base_layers=list(encoder.children())
        
        self.n_channels = in_chans
        self.n_classes = 1 # n_classes_1dec+n_classes_2dec
        
        self.patch_embed=base_layers[0]
        self.skip0=base_layers[1][0]
        self.skip1=base_layers[1][1]
        self.skip2=base_layers[1][2]
        self.center=base_layers[1][3]
        
        # first decoder
        self.deblock1_1 = block.QuadripleUp(768,384)
        self.deblock1_2 = block.QuadripleUp(384,192)
        self.deblock1_3 = block.DoubleUp(192,96)
        self.deblock1_4 = Doublewith2Up(96,48)
        self.conv1=nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock1_5 = block.DoubleUp(48,48,96)
        self.outc1 = block.OutConv(48, n_classes_1dec)
        
        # second decoder
        self.deblock2_1 = block.QuadripleUp(768,384)
        self.deblock2_2 = block.QuadripleUp(384,192)
        self.deblock2_3 = block.DoubleUp(192,96)
        self.deblock2_4 = Doublewith2Up(96,48)
        self.conv2=nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock2_5 = block.DoubleUp(48,48,96)
        self.outc2 = block.OutConv(48, n_classes_2dec)

        # third decoder
        self.deblock3_1 = block.QuadripleUp(768,384)
        self.deblock3_2 = block.QuadripleUp(384,192)
        self.deblock3_3 = block.DoubleUp(192,96)
        self.deblock3_4 = Doublewith2Up(96,48)
        self.conv3=nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock3_5 = block.DoubleUp(48,48,96)
        self.outc3 = block.OutConv(48, n_classes_3dec)
        
    def forward(self, x):
        x1=self.patch_embed(x)
        x2=self.skip0(x1)
        x3=self.skip1(x2)
        x4=self.skip2(x3)
        x5=self.center(x4)
        
        # first decoder
        out1 = self.deblock1_1(x5, x4)
        out1 = self.deblock1_2(out1, x3)
        out1 = self.deblock1_3(out1, x2)
        out1 = self.deblock1_4(out1, x1)
        out1 = self.deblock1_5(out1, self.conv1(x))
        logits1 = self.outc1(out1)
        
        # second decoder
        out2 = self.deblock2_1(x5, x4)
        out2 = self.deblock2_2(out2, x3)
        out2 = self.deblock2_3(out2, x2)
        out2 = self.deblock2_4(out2, x1)
        out2 = self.deblock2_5(out2, self.conv2(x))
        logits2 = self.outc2(out2)

        # third decoder
        out3 = self.deblock3_1(x5, x4)
        out3 = self.deblock3_2(out3, x3)
        out3 = self.deblock3_3(out3, x2)
        out3 = self.deblock3_4(out3, x1)
        out3 = self.deblock3_5(out3, self.conv3(x))
        logits3 = self.outc3(out3)
        
        return logits1, logits2, logits3