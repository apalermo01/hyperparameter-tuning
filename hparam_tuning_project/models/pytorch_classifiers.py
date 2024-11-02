import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


class TorchClassifier(nn.Module):
    """Wrapper for vanilla pytorch classifiers
    :param architecture_id:
    :param pretrained: bool - if true, loads pretrained weights (see https://pytorch.org/vision/stable/models.html#classification)
    :param num_classes
    """
    pytorch_classifier_registry = {
        'alexnet': models.alexnet,
        'vgg11': models.vgg11,
        'vgg11_bn': models.vgg11_bn,
        'vgg13': models.vgg13,
        'vgg13_bn': models.vgg13_bn,
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'vgg19': models.vgg19,
        'vgg19_bn': models.vgg19_bn,
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'squeezenet1_0': models.squeezenet1_0,
        'squeezenet1_1': models.squeezenet1_1,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet161': models.densenet161,
        'densenet201': models.densenet201,
        'inception_v3': models.inception_v3,
        'googlenet': models.googlenet,
        'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
        'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
        'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
        'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'resnext50_32x4d': models.resnext50_32x4d,
        'resnext101_32x8d': models.resnext101_32x8d,
        'wide_resnet50_2': models.wide_resnet50_2,
        'wide_resnet101_2': models.wide_resnet101_2,
        'mnasnet0_5': models.mnasnet0_5,
        'mnasnet0_75': models.mnasnet0_75,
        'mnasnet1_0': models.mnasnet1_0,
        'mnasnet1_3': models.mnasnet1_3,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7,
        'regnet_y_400mf': models.regnet_y_400mf,
        'regnet_y_800mf': models.regnet_y_800mf,
        'regnet_y_1_6gf': models.regnet_y_1_6gf,
        'regnet_y_3_2gf': models.regnet_y_3_2gf,
        'regnet_y_8gf': models.regnet_y_8gf,
        'regnet_y_16gf': models.regnet_y_16gf,
        'regnet_y_32gf': models.regnet_y_32gf,
        'regnet_y_128gf': models.regnet_y_128gf,
        'regnet_x_400mf': models.regnet_x_400mf,
        'regnet_x_800mf': models.regnet_x_800mf,
        'regnet_x_1_6gf': models.regnet_x_1_6gf,
        'regnet_x_3_2gf': models.regnet_x_3_2gf,
        'regnet_x_8gf': models.regnet_x_8gf,
        'regnet_x_16gf': models.regnet_x_16gf,
        'regnet_x_32gf': models.regnet_x_32gf,
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_l_16': models.vit_l_16,
        'vit_l_32': models.vit_l_32,
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
        'convnext_large': models.convnext_large,
    }

    def __init__(self,
                 architecture_id,
                 pretrained=True,
                 num_classes=1000,
                 num_input_channels=3,
                 **kwargs):

        super(TorchClassifier, self).__init__()
        self.model = self.pytorch_classifier_registry[architecture_id](
            pretrained=pretrained, **kwargs)

        # replace the last fully connected layer if there are a different number of classes
        if num_classes != 1000:
            old_fc = self.model._modules['fc']
            if old_fc.bias is None:
                bias = False
            else:
                bias = True
            self.model._modules['fc'] = nn.Linear(in_features=old_fc.in_features,
                                                  out_features=num_classes,
                                                  bias=bias)

        # replace input convolutional layer if there are not 3 input channels
        if num_input_channels != 3:

            if 'conv1' in self.model._modules:
                old_input_conv = self.model._modules['conv1']
                if old_input_conv.bias is None:
                    bias = False
                else:
                    bias = True

                self.model._modules['conv1'] = nn.Conv2d(
                    in_channels=num_input_channels,
                    out_channels=old_input_conv.out_channels,
                    kernel_size=old_input_conv.kernel_size,
                    stride=old_input_conv.stride,
                    padding=old_input_conv.padding,
                    dilation=old_input_conv.dilation,
                    groups=old_input_conv.groups,
                    bias=bias,
                    padding_mode=old_input_conv.padding_mode,
                )

            # regnet
            elif 'stem' in self.model._modules:
                old_input_conv = self.model._modules['stem']._modules['0']
                if old_input_conv.bias is None:
                    bias = False
                else:
                    bias = True

                self.model._modules['stem']._modules['0'] = nn.Conv2d(
                    in_channels=num_input_channels,
                    out_channels=old_input_conv.out_channels,
                    kernel_size=old_input_conv.kernel_size,
                    stride=old_input_conv.stride,
                    padding=old_input_conv.padding,
                    dilation=old_input_conv.dilation,
                    groups=old_input_conv.groups,
                    bias=bias,
                    padding_mode=old_input_conv.padding_mode,
                )

        logger.info(f"model loaded: {architecture_id}")

    def forward(self, x):
        return self.model(x)
