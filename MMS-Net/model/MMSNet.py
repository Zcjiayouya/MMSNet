import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(DoubleConv,self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate),
            # CBAM(gate_channels=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):

        xout=self.conv(x)
        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class SCM7(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        factor_num=16
        nb_filter=[1*factor_num,2*factor_num,4*factor_num,8*factor_num,16*factor_num]

        self.pool=nn.MaxPool2d(2,2)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.conv0_0=DoubleConv(in_channel,nb_filter[0])
        self.conv1_0=DoubleConv(nb_filter[0],nb_filter[1])
        self.conv2_0=DoubleConv(nb_filter[1],nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1=DoubleConv(nb_filter[0]+nb_filter[1],nb_filter[0])
        self.conv1_1=DoubleConv(nb_filter[1]+nb_filter[2],nb_filter[1])
        self.conv2_1=DoubleConv(nb_filter[2]+nb_filter[3],nb_filter[2])
        self.conv3_1=DoubleConv(nb_filter[3]+nb_filter[4],nb_filter[3])

        self.conv0_2=DoubleConv(nb_filter[0]*2+nb_filter[1],nb_filter[0])
        self.conv1_2=DoubleConv(nb_filter[1]*2+nb_filter[2],nb_filter[1])
        self.conv2_2=DoubleConv(nb_filter[2]*2+nb_filter[3],nb_filter[2])

        self.conv0_3=DoubleConv(nb_filter[0]*3+nb_filter[1],nb_filter[0])
        self.conv1_3=DoubleConv(nb_filter[1]*3+nb_filter[2],nb_filter[1])

        self.conv0_4=DoubleConv(nb_filter[0]*4+nb_filter[1],nb_filter[0])
        self.sigmoid=nn.Sigmoid()

        self.final1=nn.Conv2d(nb_filter[0],out_channel,kernel_size=1)
        self.final2=nn.Conv2d(nb_filter[0],out_channel,kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
    def forward(self,input):
        x0_0=self.conv0_0(input)
        x1_0=self.conv1_0(self.pool(x0_0))
        x0_1=self.conv0_1(torch.cat([x0_0,self.up(x1_0)],1))

        x2_0=self.conv2_0(self.pool(x1_0))
        x1_1=self.conv1_1(torch.cat([x1_0,self.up(x2_0)],1))
        x0_2=self.conv0_2(torch.cat([x0_0,x0_1,self.up(x1_1)],1))

        x3_0=self.conv3_0(self.pool(x2_0))
        x2_1=self.conv2_1(torch.cat([x2_0,self.up(x3_0)],1))
        x1_2=self.conv1_2(torch.cat([x1_0,x1_1,self.up(x2_1)],1))
        x0_3=self.conv0_3(torch.cat([x0_0,x0_1,x0_2,self.up(x1_2)],1))

        x4_0=self.conv4_0(torch.cat([self.pool(x3_0)],1))
        x3_1=self.conv3_1(torch.cat([x3_0,self.up(x4_0)],1))
        x2_2=self.conv2_2(torch.cat([x2_0,x2_1,self.up(x3_1)],1))
        x1_3=self.conv1_3(torch.cat([x1_0,x1_1,x1_2,self.up(x2_2)],1))
        x0_4=self.conv0_4(torch.cat([x0_0,x0_1,x0_2,x0_3,self.up(x1_3)],1))

        output1=self.sigmoid(self.final1(x0_1))
        output2 = self.sigmoid(self.final2(x0_2))
        output3 = self.sigmoid(self.final3(x0_3))
        output4 = self.sigmoid(self.final4(x0_4))

        # return 0.7*output1+0.1*output2+0.1*output3+0.1*output4
        return  (output1 + output2 +  output3 +  output4)/4


class SCM6(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        factor_num = 32
        nb_filter = [1 * factor_num, 2 * factor_num, 4 * factor_num, 8 * factor_num, 16 * factor_num]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])


        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])

        self.sigmoid = nn.Sigmoid()

        self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))


        output1 = self.sigmoid(self.final1(x0_1))
        output2 = self.sigmoid(self.final2(x0_2))
        output3 = self.sigmoid(self.final3(x0_3))

        return (output1 + output2 +  output3 )/3

### RSU-5 ###
class SCM5(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        factor_num = 16
        nb_filter = [1 * factor_num, 2 * factor_num, 4 * factor_num, 8 * factor_num, 16 * factor_num]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])

        self.sigmoid = nn.Sigmoid()

        self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        output1 = self.sigmoid(self.final1(x0_1))
        output2 = self.sigmoid(self.final2(x0_2))

        return (output1 + output2 )/2

### RSU-4 ###
class SCM4(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        factor_num = 32
        nb_filter = [1 * factor_num, 2 * factor_num, 4 * factor_num, 8 * factor_num, 16 * factor_num]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.sigmoid = nn.Sigmoid()

        self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        output1 = self.sigmoid(self.final1(x0_1))

        return output1

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = DoubleConv(in_ch,out_ch,dirate=1)

        self.rebnconv1 = DoubleConv(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = DoubleConv(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = DoubleConv(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = DoubleConv(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = DoubleConv(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = DoubleConv(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = DoubleConv(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

class MMSNet(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(MMSNet, self).__init__()

        self.stage1 = SCM6(in_ch, 64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = SCM6(64, 128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = SCM5(128, 256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = SCM4(256, 512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = SCM4(1024, 256)
        self.stage3d = SCM5(512, 128)
        self.stage2d = SCM6(256, 64)
        self.stage1d = SCM6(128, 64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

