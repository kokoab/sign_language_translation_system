import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
                                      3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.stack(eye_array).clone().detach().to('cuda').requires_grad_(False), requires_grad=False)

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        # Adjust input dimensions from [batch_size, num_point, num_channels] to [batch_size, num_channels, 1, num_point]

        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x

class GCNClassifier(nn.Module):
    def __init__(self, num_class=10, num_point=11, num_channels=3, groups=8, graph=None, graph_args=dict()):
        super(GCNClassifier, self).__init__()

        if graph is None:
            raise ValueError("Graph is required")
        else:
            Graph = self.import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.gcn1 = unit_gcn(num_channels, 64, A, groups, num_point)
        self.gcn2 = unit_gcn(64, 128, A, groups, num_point)
        self.gcn3 = unit_gcn(128, 256, A, groups, num_point)

        self.fc = nn.Linear(256, num_class)

    @staticmethod
    def import_class(name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def forward(self, x):
        # Input x: [batch_size, num_point, num_channels]
        # Adjust dimensions to [batch_size, num_channels, 1, num_point] for processing
        x = x.permute(0, 2, 1).unsqueeze(2)

        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)

        # Global average pooling
        x = x.mean(dim=-1).mean(dim=-1)  # [batch_size, 256]

        # Classification
        x = self.fc(x)  # [batch_size, num_class]
        return x
