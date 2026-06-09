import torch
from ultralytics.nn.modules import Conv, C3


class C3k2(torch.nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.cv1 = Conv(c1, c2, k=1)
        self.cv2 = Conv(c1, c2, k=3)
        self.m = torch.nn.Sequential(*(C3(c2, c2, n, shortcut, g, e) for _ in range(n)))

    def forward(self, x):
        return self.m(torch.cat((self.cv1(x), self.cv2(x)), dim=1))
