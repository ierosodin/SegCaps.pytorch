import torch
import torch.nn as nn
from models.capsule_layer import CapsuleLayer
import models.nn_


class SegCaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, padding=2),
        )
        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1, 16, "conv", kernel_size=5, stride=2, out_num_capsules=2, out_dims=16, routing=1),
            CapsuleLayer(2, 16, "conv", kernel_size=5, stride=1, out_num_capsules=4, out_dims=16, routing=3),
        )
        self.step_2 = nn.Sequential(  # 1/4
            CapsuleLayer(4, 16, "conv", kernel_size=5, stride=2, out_num_capsules=4, out_dims=32, routing=3),
            CapsuleLayer(4, 32, "conv", kernel_size=5, stride=1, out_num_capsules=8, out_dims=32, routing=3)
        )
        self.step_3 = nn.Sequential(  # 1/8
            CapsuleLayer(8, 32, "conv", kernel_size=5, stride=2, out_num_capsules=8, out_dims=64, routing=3),
            CapsuleLayer(8, 64, "conv", kernel_size=5, stride=1, out_num_capsules=8, out_dims=32, routing=3)
        )
        self.step_4 = CapsuleLayer(8, 32, "deconv", kernel_size=5, stride=2, out_num_capsules=8, out_dims=32, routing=3)

        self.step_5 = CapsuleLayer(16, 32, "conv", kernel_size=5, stride=1, out_num_capsules=4, out_dims=32, routing=3)

        self.step_6 = CapsuleLayer(4, 32, "deconv", kernel_size=5, stride=2, out_num_capsules=4, out_dims=16, routing=3)
        self.step_7 = CapsuleLayer(8, 16, "conv", kernel_size=5, stride=1, out_num_capsules=4, out_dims=16, routing=3)
        self.step_8 = CapsuleLayer(4, 16, "deconv", kernel_size=5, stride=2, out_num_capsules=2, out_dims=16, routing=3)
        self.step_10 = CapsuleLayer(3, 16, "conv", kernel_size=5, stride=1, out_num_capsules=1, out_dims=16, routing=3)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 1, 5, 1, padding=2),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x.unsqueeze_(1)

        skip_1 = x  # [N,1,16,H,W]

        x = self.step_1(x)

        skip_2 = x  # [N,4,16,H/2,W/2]
        x = self.step_2(x)

        skip_3 = x  # [N,8,32,H/4,W/4]

        x = self.step_3(x)  # [N,8,32,H/8,W/8]

        x = self.step_4(x)  # [N,8,32,H/4,W/4]
        x = torch.cat((x, skip_3), 1)  # [N,16,32,H/4,W/4]

        x = self.step_5(x)  # [N,4,32,H/4,W/4]

        x = self.step_6(x)  # [N,4,16,H/2,W/2]

        x = torch.cat((x, skip_2), 1)   # [N,8,16,H/2,W/2]
        x = self.step_7(x)  # [N,4,16,H/2,W/2]
        x = self.step_8(x)  # [N,2,16,H,W]

        x = torch.cat((x, skip_1), 1)
        x = self.step_10(x)
        x.squeeze_(1)
        v_lens = torch.norm(x, p=2, dim=1)
        return v_lens

