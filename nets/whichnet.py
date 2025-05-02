from nets import uNet, AttNet, MTuNet, swinv2Unet , SwinUMamba, SwinUMambaD
from nets.vit_seg_modeling import VisionTransformer as ViT_seg
from nets.vit_seg_modeling import MTVisionTransformer as MTViT_seg
from nets.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from nets.MedT import gated, MTgated
from nets.segmenter import load_model_segmenter, MTload_model_segmenter
from nets.AE import CAEDoubleConv
import numpy as np

def whichnet(net_id, n_classes, img_size=None):
    
    if net_id == 1:
        vgg = False
        net = uNet.uNet(n_channels = 1, n_classes = n_classes)
        
    elif net_id == 2:
        pretrained = False
        vgg = True
        net = uNet.uNet13(n_classes = n_classes, pretrained = pretrained)

    elif net_id == 3:
        pretrained = True
        vgg = True
        net = uNet.uNet13(n_classes = n_classes, pretrained = pretrained)   

    elif net_id == 4:
        pretrained = False
        vgg = True
        net = uNet.uNet16(n_classes = n_classes, pretrained = pretrained)

    elif net_id == 5:
        pretrained = True
        vgg = True
        net = uNet.uNet16(n_classes = n_classes, pretrained = pretrained)

    elif net_id == 6:
        pretrained = False
        vgg = True
        net = uNet.uNet19(n_classes = n_classes, pretrained = pretrained)

    elif net_id == 7:
        pretrained = True
        vgg = True
        net = uNet.uNet19(n_classes = n_classes, pretrained = pretrained)
        
    elif net_id == 8:
        pretrained, vgg, attention = False, True, False
        net = AttNet.uNet13Att(n_classes = n_classes, pretrained = pretrained, attention = attention)
        
    elif net_id == 9:
        pretrained, vgg, attention = True, True, False
        net = AttNet.uNet13Att(n_classes = n_classes, pretrained = pretrained, attention = attention)
        
    elif net_id == 10:
        pretrained, vgg, attention = True, True, True
        net = AttNet.uNet13Att(n_classes = n_classes, pretrained = pretrained, attention = attention)  

    elif net_id == 11:
        pretrained, vgg, attention = False, True, False
        net = AttNet.uNet16Att(n_classes = n_classes, pretrained = pretrained, attention = attention)
        
    elif net_id == 12:
        pretrained, vgg, attention = True, True, False
        net = AttNet.uNet16Att(n_classes = n_classes, pretrained = pretrained, attention = attention)
        
    elif net_id == 13:
        pretrained, vgg, attention = True, True, True
        net = AttNet.uNet16Att(n_classes = n_classes, pretrained = pretrained, attention = attention)
        
    elif net_id == 14:
        pretrained, vgg, attention = False, True, False
        net = AttNet.uNet19Att(n_classes = n_classes, pretrained = pretrained, attention = attention)
        
    elif net_id == 15:
        pretrained, vgg, attention = True, True, False
        net = AttNet.uNet19Att(n_classes = n_classes, pretrained = pretrained, attention = attention)
        
    elif net_id == 16:
        pretrained, vgg, attention = True, True, True
        net = AttNet.uNet19Att(n_classes = n_classes, pretrained = pretrained, attention = attention)  
        
    elif net_id == 17:
        pretrained = True
        vgg = True
        net = MTuNet.MTuNet19(n_classes = n_classes, pretrained = pretrained)
        
    elif net_id == 18:
        pretrained = True
        vgg = True
        net = MTuNet.MTcuNet19(n_classes = n_classes, pretrained = pretrained)
        
    elif net_id == 19:
        vit_name = 'R50-ViT-B_16'
        vgg = True
        vit_patches_size = 16
        n_skip = 3
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = n_classes
        config_vit.n_skip = n_skip
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        net = ViT_seg(config_vit, img_size=img_size, n_classes=config_vit.n_classes).cuda()
        net.load_from(weights=np.load(config_vit.pretrained_path))
        
    elif net_id == 20:
        vgg = False
        net = gated(img_size=img_size, imgchan=1, num_classes=n_classes).cuda()
        net.n_channels = 1
        net.n_classes = n_classes
        
    elif net_id == 21:
        vgg = True
        pretrained = False # NEW!
        net = load_model_segmenter(model_path='./vit_checkpoint/', pretrained=pretrained, nb_channel=3, nb_class=n_classes).cuda()
        net.n_channels = 3
        net.n_classes = n_classes
        
    elif net_id == 22:
        vgg = False
        net = MTgated(img_size=img_size, imgchan=1, num_classes=n_classes).cuda()
        net.n_channels = 1
        net.n_classes = n_classes

    elif net_id == 23:
        vit_name = 'R50-ViT-B_16'
        vgg = True
        vit_patches_size = 16
        n_skip = 3
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = n_classes
        config_vit.n_skip = n_skip
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        net = MTViT_seg(config_vit, img_size=img_size, n_classes=config_vit.n_classes).cuda()
        net.load_from(weights=np.load(config_vit.pretrained_path))
            
    elif net_id == 24:
        pretrained = False
        vgg = False
        net = CAEDoubleConv(n_classes = n_classes)
        
    elif net_id == 25:
        vgg = True
        pretrained = False # NEW!
        net = MTload_model_segmenter(model_path='./vit_checkpoint/', pretrained=pretrained, nb_channel=3, nb_class=n_classes).cuda()
        net.n_channels = 3
        net.n_classes = n_classes
    
    elif net_id == 26:
        net=swinv2Unet.SwinV2OneDecoder(model_name='swinv2_cr_tiny_ns_224',
                 pretrained=True,
                 in_chans=1,
                 img_size=(img_size,img_size),
                 patch_size=4,
                 n_classes=n_classes)
        vgg = False
        
    elif net_id == 27:
        net=swinv2Unet.SwinV2TwoDecoder(model_name='swinv2_cr_tiny_ns_224',
                 pretrained=True,
                 img_size=(img_size,img_size),
                 in_chans=1,
                 n_classes_1dec=1,
                 n_classes_2dec=1)
        vgg = False

    elif net_id == 28:
        net=swinv2Unet.SwinV2ThreeDecoder(model_name='swinv2_cr_tiny_ns_224',
                 pretrained=True,
                 img_size=(img_size,img_size),
                 in_chans=1,
                 n_classes_1dec=1,
                 n_classes_2dec=1,
                 n_classes_3dec=1)
        vgg = False
    #"""
    elif net_id == 29:
        net = SwinUMamba.SwinUMamba(in_chans=1, out_chans=n_classes).cuda()
        # net = SwinUMamba.load_pretrained_ckpt(net)
        # from torchsummary import summary
        # summary(net, (1, 128, 128))
        vgg = False
        net.n_channels = 1
        net.n_classes = n_classes
    #"""
    
    return net, vgg