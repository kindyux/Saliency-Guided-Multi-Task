from nets.CSPdarknet import C3, Conv
from nets.swin_transformer import *


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody_Salience_Swin(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody_Salience_Swin, self).__init__()

        self.swin_dim = 96
        self.backbone = swin_tiny_patch4_window7_224()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.conv_for_feat3_half = Conv(self.swin_dim * 8, self.swin_dim * 4, 1, 1)
        self.conv3_for_upsample1 = C3(self.swin_dim * 8, self.swin_dim * 4, n=4, shortcut=True)

        self.conv_for_feat2_half = Conv(self.swin_dim * 4, self.swin_dim * 2, 1, 1)
        self.conv3_for_upsample2 = C3(self.swin_dim * 4, self.swin_dim * 2, n=4, shortcut=True)

        self.conv_for_feat1_half = Conv(self.swin_dim * 2, self.swin_dim * 1, 1, 1)
        self.conv3_for_upsample0 = C3(self.swin_dim * 2, self.swin_dim * 1, n=4, shortcut=True)

        # ---------------------------------------------------#
        # ---------------------------------------------------#
        # ---------------------------------------------------#

        self.down_sample_0 = Conv(self.swin_dim * 1, self.swin_dim * 1, 3, 2)
        self.conv3_for_downsample_0 = C3(self.swin_dim * 2, self.swin_dim * 2, n=4, shortcut=True)

        self.down_sample0 = Conv(self.swin_dim * 2, self.swin_dim * 2, 3, 2)
        self.conv3_for_downsample0 = C3(self.swin_dim * 4, self.swin_dim * 4, n=4, shortcut=True)

        self.down_sample1 = Conv(self.swin_dim * 4, self.swin_dim * 4, 3, 2)
        self.conv3_for_downsample1 = C3(self.swin_dim * 8, self.swin_dim * 8, n=4, shortcut=True)

        # ---------------------------------------------------#
        # ---------------------------------------------------#
        # ---------------------------------------------------#

        self.salience = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear"),
            C3(self.swin_dim * 1, int(self.swin_dim / 2), n=2, shortcut=True),
            nn.Conv2d(int(self.swin_dim / 2), 8, 1, 1, bias=False),
            nn.Conv2d(8, 1, 1, 1, bias=False)
        )

        self.yolo_head_P2 = nn.Conv2d(self.swin_dim * 2, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P3 = nn.Conv2d(self.swin_dim * 4, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(self.swin_dim * 8, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        feat1, feat2, feat3, feat4 = self.backbone(x)

        P4 = self.conv_for_feat3_half(feat4)  # CBS降维
        P4_upsample = self.upsample(P4)     # 上采样
        P3 = torch.cat([P4_upsample, feat3], dim=1)  # 拼接

        P3 = self.conv3_for_upsample1(P3)  # 利用C3将拼接后的结果

        P3 = self.conv_for_feat2_half(P3)
        P3_upsample = self.upsample(P3)
        P2 = torch.cat([P3_upsample, feat2], dim=1)
        P2 = self.conv3_for_upsample2(P2)

        P2 = self.conv_for_feat1_half(P2)
        P2_upsample = self.upsample(P2)
        P1 = torch.cat([P2_upsample, feat1], dim=1)
        # 128*104*104 --> 64*104*104
        P1 = self.conv3_for_upsample0(P1)

        salience = torch.sigmoid(self.salience(P1))

        P1_downsample = self.down_sample_0(P1)
        P2 = torch.cat([P1_downsample, P2], dim=1)

        P2 = self.conv3_for_downsample_0(P2)

        P2_downsample = self.down_sample0(P2)
        P3 = torch.cat([P2_downsample, P3], dim=1)
        P3 = self.conv3_for_downsample0(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], dim=1)
        P4 = self.conv3_for_downsample1(P4)

        out2 = self.yolo_head_P2(P2)

        out1 = self.yolo_head_P3(P3)

        out0 = self.yolo_head_P4(P4)

        return out0, out1, out2, salience

    def shared_modules(self):
        return [self.backbone,
                self.conv_for_feat3_half,
                self.conv3_for_upsample1,
                self.conv_for_feat2_half,
                self.conv3_for_upsample2,
                self.conv_for_feat1_half,
                self.conv3_for_upsample0
                ]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def filt_trainable_params(self):
        return [param for param in self.parameters() if param.requires_grad]


if __name__ == '__main__':
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 1
    model = YoloBody_Salience_Swin(anchors_mask, num_classes)

    # from torchstat import stat
    # stat(model, (3, 512, 512))

    # input = torch.ones((2, 1, 512, 512))
    # output = model(input)
    # for i in output:
    #     print(i.shape)

    import torch
    from thop import profile

    input = torch.randn(1, 3, 416, 416)
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)