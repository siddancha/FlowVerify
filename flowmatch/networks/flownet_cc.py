'''
Original code from: https://github.com/NVIDIA/flownet2-pytorch
Modified by: Siddharth Ancha

Portions of this code copyright 2017, Clement Pinard
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from flowmatch.networks import submodules
from flowmatch.networks.flownet import FlowNet

'Parameter count, 39, 175, 298'


class FlowNetC(FlowNet):
    def __init__(self, batchNorm=True, cfg=None):
        super(FlowNetC, self).__init__(cfg)

        self.batchNorm = batchNorm
        # self.div_flow = div_flow

        if self.cfg.coord_conv:
            conv = submodules.coord_conv
            deconv = submodules.coord_deconv
            predict_flow = submodules.predict_flow_coord
            conv_transpose = submodules.CoordConvTranspose2d
        else:
            conv = submodules.conv
            deconv = submodules.deconv
            predict_flow = submodules.predict_flow
            conv_transpose = nn.ConvTranspose2d

        # First part of FlowNetC preceding cross-correlation layer. Reduces spatial dimension by 8.
        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        # self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2,
        #                         corr_multiply=1)
        # self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        # self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv3_1 = conv(self.batchNorm, 32*32 + 32, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = conv_transpose(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = conv_transpose(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = conv_transpose(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = conv_transpose(2, 2, 4, 2, 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

        self.upsample = lambda x, scale_factor:\
            F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, cs_input, tg_input, collect_summaries=False):
        out_conv1a = self.conv1(tg_input)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(cs_input)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams: perform cross-correlation between every pair of pixels.
        # out_corr = self.corr(out_conv3a, out_conv3b)  # False
        # TODO: Figure out why corr_activation being used if correlation is nothing but a dot-product.
        # out_corr = self.corr_activation(out_corr)
        assert(out_conv3a.shape[:2] == out_conv3b.shape[:2])
        b, c = out_conv3a.shape[:2]
        ha, wa = out_conv3a.shape[2:]
        hb, wb = out_conv3b.shape[2:]
        # Shape of expanded conv3a and conv3b should be [b, c, hb * wb, ha, wa].
        expanded_conv3a = out_conv3a.unsqueeze(2).expand(b, c, hb * wb, ha, wa)
        expanded_conv3b = out_conv3b.view(b, c, hb * wb, 1, 1).expand(b, c, hb * wb, ha, wa)
        out_corr = (expanded_conv3a * expanded_conv3b).mean(dim=1)  # new number of feature channels are hb * wb

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)

        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)

        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)

        flow2 = self.predict_flow2(concat2)

        if collect_summaries:
            list_tags = ['out_conv1a', 'out_conv2a', 'out_conv3a',
                         'out_conv1b', 'out_conv2b', 'out_conv3b',
                         'out_corr', 'out_conv_redir',
                         'out_conv3_1', 'out_conv4', 'out_conv5', 'out_conv6',
                         'flow6', 'flow5', 'flow4', 'flow3', 'flow2',
                         'out_deconv5', 'out_deconv4', 'out_deconv3', 'out_deconv2'
                         ]
            list_tensors = [out_conv1a, out_conv2a, out_conv3a,
                            out_conv1b, out_conv2b, out_conv3b,
                            out_corr, out_conv_redir,
                            out_conv3_1, out_conv4, out_conv5, out_conv6,
                            flow6, flow5, flow4, flow3, flow2,
                            out_deconv5, out_deconv4, out_deconv3, out_deconv2
                         ]
            summaries = zip(list_tags, [t.detach().cpu().numpy() for t in list_tensors])
        else:
            summaries = None

        # List of (flow_i, stride_i) tuples. flow_i has stride stride_i and needs to be upsampled by stride_i to match
        # the stride of the input.
        pred_flows_and_strides = [(flow2, 4), (flow3, 8), (flow4, 16), (flow5, 32), (flow6, 64)]
        return pred_flows_and_strides, summaries
