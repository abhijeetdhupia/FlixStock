import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class OutputModel(nn.Module):
    def __init__(self, n_neck_classes, n_sleeve_classes, n_pattern_classes):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.neck = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_neck_classes)
        )
        self.sleeve = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_sleeve_classes)
        )
        self.pattern = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_pattern_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'neck': self.neck(x),
            'sleeve': self.sleeve(x),
            'pattern': self.pattern(x)
        }

    def get_loss(self, net_output, ground_truth):
        neck_loss = F.cross_entropy(net_output['neck'], ground_truth['neck_labels'])
        sleeve_loss = F.cross_entropy(net_output['sleeve'], ground_truth['sleeve_labels'])
        pattern_loss = F.cross_entropy(net_output['pattern'], ground_truth['pattern_labels'])
        loss = neck_loss + sleeve_loss + pattern_loss
        return loss, {'neck': neck_loss, 'sleeve': sleeve_loss, 'pattern': pattern_loss}