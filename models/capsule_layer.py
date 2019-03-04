import torch
import torch.nn as nn
import torch.nn.functional as F
import models.nn_ as nn_
import torch.optim as optim

class CapsuleLayer(nn.Module):
    def __init__(self, in_num_capsules, in_dims, op, kernel_size, stride, out_num_capsules, out_dims, routing):
        super().__init__()
        self.out_num_capsules = out_num_capsules
        self.out_dims = out_dims
        self.op = op
        self.kernel_size = kernel_size
        self.stride = stride
        self.routing = routing
        self.in_num_capsules = in_num_capsules
        if self.op == 'conv':
            self.convs = nn.ModuleList(
                [nn.Conv2d(in_dims, out_num_capsules * out_dims, kernel_size=kernel_size, stride=stride, padding=2) for _ in range(in_num_capsules)])
        else:
            self.convs = nn.ModuleList(
                [nn.ConvTranspose2d(in_dims, out_num_capsules * out_dims, kernel_size=kernel_size, stride=stride, padding=2, output_padding=1) for _ in range(in_num_capsules)])

    def forward(self, u):  # input [N,CAPS,C,H,W]
        if u.shape[1] != self.in_num_capsules:
            raise ValueError("Wrong type of operation for capsule")
        self.batch_size = u.shape[0]

        u_t_list = [u_t.squeeze(1) for u_t in u.split(1, 1)]  # extract capsule
        u_hat_t_list = []

        for i, u_t in enumerate(u_t_list):  # u_t: [N,C,H,W]
            if self.op == "conv":
                u_hat_t = self.convs[i](u_t)
            elif self.op == "deconv":
                u_hat_t = self.convs[i](u_t)  # u_hat_t: [N,t_1*z_1,H,W]
            else:
                raise ValueError("Wrong type of operation for capsule")
            h = u_hat_t.shape[2]
            w = u_hat_t.shape[3]
            u_hat_t = u_hat_t.reshape(self.batch_size, self.out_num_capsules, self.out_dims, h, w).transpose_(1, 3).transpose_(2, 4)
            u_hat_t_list.append(u_hat_t)  # [N,H_1,W_1,t_1,z_1]
        v = self.update_routing(u_hat_t_list, h, w)
        return v

    def update_routing(self, u_hat_t_list, h, w):
        one_kernel = torch.ones(1, self.out_num_capsules, self.kernel_size, self.kernel_size).cuda()  # no need to update
        b = torch.zeros(self.batch_size, h, w, self.in_num_capsules, self.out_num_capsules).cuda()  # no need to update
        b_t_list = [b_t.squeeze(3) for b_t in b.split(1, 3)]

        for d in range(self.routing):
            r_t_mul_u_hat_t_list = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list):
                # routing softmax (N,H_1,W_1,t_1)
                b_t.transpose_(1, 3).transpose_(2, 3)  # [N,t_1,H_1, W_1]
                b_t_max = torch.nn.functional.max_pool2d(b_t, self.kernel_size, 1, padding=2)
                b_t_max = b_t_max.max(1, True)[0]
                c_t = torch.exp(b_t - b_t_max)
                sum_c_t = nn_.conv2d_same(c_t, one_kernel, stride=(1, 1))  # [... , 1]
                r_t = c_t / sum_c_t  # [N,t_1, H_1, W_1]
                r_t = r_t.transpose(1, 3).transpose(1, 2)  # [N, H_1, W_1,t_1]
                r_t = r_t.unsqueeze(4)  # [N, H_1, W_1,t_1, 1]
                r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, H_1, W_1, t_1, z_1]
            p = sum(r_t_mul_u_hat_t_list)  # [N, H_1, W_1, t_1, z_1]
            v = self.squash(p)
            if d != self.routing - 1:
                b_t_list_ = []
                for b_t, u_hat_t in zip(b_t_list, u_hat_t_list):
                    # b_t     : [N, t_1,H_1, W_1]
                    # u_hat_t : [N, H_1, W_1, t_1, z_1]
                    # v       : [N, H_1, W_1, t_1, z_1]
                    # [N,H_1,W_1,t_1]
                    b_t.transpose_(1,3).transpose_(2,1)
                    b_t_list_.append(b_t + (u_hat_t * v).sum(4))
        v.transpose_(1, 3).transpose_(2, 4)
        return v

    def squash(self, p):
        p_norm_sq = (p * p).sum(-1, True)
        p_norm = (p_norm_sq + 1e-9).sqrt()
        v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
        return v

