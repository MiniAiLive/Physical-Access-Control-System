from torch import nn


class FASNetB(nn.Module):
    """ define the model called FASNetB """
    def __init__(self, scale=1.0, expand_ratio=1):
        super(FASNetB, self).__init__()
        def conv_bn(inp, oup, stride = 1):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.PReLU(oup)
            )
        def conv_dw(inp, oup, stride = 1):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.PReLU(inp),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.PReLU(oup),
            )
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout()
        self.head = conv_bn(3, (int)(32 * scale))
        self.step1 = nn.Sequential(
            conv_dw((int)(32 * scale), (int)(64 * scale), 2),
            conv_dw((int)(64 * scale), (int)(128 * scale)),
            conv_dw((int)(128 * scale), (int)(128 * scale)),
        )
        self.step1_shotcut = conv_dw((int)(32 * scale), (int)(128 * scale), 2)

        self.step2 = nn.Sequential(
            conv_dw((int)(128 * scale), (int)(128 * scale), 2),
            conv_dw((int)(128 * scale), (int)(256 * scale)),
            conv_dw((int)(256 * scale), (int)(256 * scale)),
        )
        self.step2_shotcut = conv_dw((int)(128 * scale), (int)(256 * scale), 2)
        self.depth_ret = nn.Sequential(
            nn.Conv2d((int)(256 * scale), (int)(256 * scale), 3, 1, 1, groups=(int)(256 * scale), bias=False),
            nn.BatchNorm2d((int)(256 * scale)),
            nn.Conv2d((int)(256 * scale), 2, 1, 1, 0, bias=False),
        )
        self.depth_shotcut = conv_dw((int)(256 * scale), 2)
        self.class_ret = nn.Linear(2048, 3)

    def forward(self, x):
        """ forward model with the input image """
        head = self.head(x)
        step1 = self.step1(head) + self.step1_shotcut(head)
        step2 = self.dropout(self.step2(step1) + self.step2_shotcut(step1))
        depth = self.softmax(self.depth_ret(step2))
        class_pre = self.depth_shotcut(step2) + depth
        class_pre = class_pre.view(-1, 2048)
        class_ret = self.class_ret(class_pre)
        return class_ret
