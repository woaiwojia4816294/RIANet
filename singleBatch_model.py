import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(4, 4))
        self.pool3 = nn.AvgPool2d(kernel_size=(9, 9))

        self.model_attention = nn.Conv2d(96, 10, kernel_size=1, stride=1, padding=0,
                                         bias=nn.InstanceNorm2d)
        self.model_image = nn.Conv2d(96, 10 * 3, kernel_size=3, stride=1, padding=1,
                                     bias=nn.InstanceNorm2d)

    def forward(self, x):  # x.shape 2,32,240,480
        pool_feature1 = self.pool1(x)  # 2,32,240,480
        pool_feature2 = self.pool2(x)  # 2,32,60,120
        pool_feature3 = self.pool3(x)  # 2,32,26,53

        pool_feature1_up = F.interpolate(input=pool_feature1, size=(len(x[0, 0, :, 0]), len(x[0, 0, 0, :])),
                                         mode='bilinear', align_corners=True)  # 2,32,240,320
        pool_feature2_up = F.interpolate(input=pool_feature2, size=(len(x[0, 0, :, 0]), len(x[0, 0, 0, :])),
                                         mode='bilinear', align_corners=True)
        pool_feature3_up = F.interpolate(input=pool_feature3, size=(len(x[0, 0, :, 0]), len(x[0, 0, 0, :])),
                                         mode='bilinear', align_corners=True)

        feature_combine = torch.cat((pool_feature1_up, pool_feature2_up, pool_feature3_up), 1)  # 2,96,240,480
        attention = self.model_attention(feature_combine)  # 2,10,240,480
        image = self.model_image(feature_combine)  # 2,30,240,480
        # attention 1,10,240,480
        img = []
        for i in range(len(attention[:, 0, 0, 0])):
            attention1 = attention[i, 0, :, :].repeat(1, 3, 1, 1)  # 1,3,240,480
            attention2 = attention[i, 1, :, :].repeat(1, 3, 1, 1)
            attention3 = attention[i, 2, :, :].repeat(1, 3, 1, 1)
            attention4 = attention[i, 3, :, :].repeat(1, 3, 1, 1)
            attention5 = attention[i, 4, :, :].repeat(1, 3, 1, 1)
            attention6 = attention[i, 5, :, :].repeat(1, 3, 1, 1)
            attention7 = attention[i, 6, :, :].repeat(1, 3, 1, 1)
            attention8 = attention[i, 7, :, :].repeat(1, 3, 1, 1)
            attention9 = attention[i, 8, :, :].repeat(1, 3, 1, 1)
            attention10 = attention[i, 9, :, :].repeat(1, 3, 1, 1)
            # plt.imshow(np.transpose(attention10.squeeze(0).cpu().detach().numpy(),(1,2,0)))
            # plt.show()
            # 65536
            image1 = image[i, 0:3, :, :]
            image2 = image[i, 3:6, :, :]
            image3 = image[i, 6:9, :, :]
            image4 = image[i, 9:12, :, :]
            image5 = image[i, 12:15, :, :]
            image6 = image[i, 15:18, :, :]
            image7 = image[i, 18:21, :, :]
            image8 = image[i, 21:24, :, :]
            image9 = image[i, 24:27, :, :]
            image10 = image[i, 27:30, :, :]

            output1 = image1 * attention1  # 2,3,240,480 * 1,6,240,480
            output2 = image2 * attention2
            output3 = image3 * attention3
            output4 = image4 * attention4
            output5 = image5 * attention5
            output6 = image6 * attention6
            output7 = image7 * attention7
            output8 = image8 * attention8
            output9 = image9 * attention9
            output10 = image10 * attention10

            output = output1 + output2 + output3 + output4 + output5 + \
                     output6 + output7 + output8 + output9 + output10
            img.append(output)
        return img
        # return output


class AttenNet(nn.Module):
    def __init__(self, fem_channel=32, input_channel=8, block_num=9):
        super(AttenNet, self).__init__()
        self.block_num = block_num
        self.FEM = nn.Conv2d(3, fem_channel, (3, 3), padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.attention = AttentionModule()

        blocks = []
        for _ in range(self.block_num):  # 9
            blocks += [nn.Conv2d(fem_channel, fem_channel, (3, 3), padding=1)]
            blocks += [AttentionModule()]

        self.blocks = nn.Sequential(*blocks)
        self.FM = nn.Conv2d(3 * (self.block_num + 1), 3, (1, 1))

    def forward(self, x):
        x = self.FEM(x)
        x = self.activation(x)
        atten_features = []
        atten_features.append(self.attention(x))

        for i in range(self.block_num):
            x = self.blocks[2 * i](x)
            atten_features.append(self.blocks[2 * i + 1](x))

        for index in range(len(atten_features[0])):
            atten_feature_batch = torch.cat(list(zip(*atten_features))[index], dim=1)
            final_img = self.FM(atten_feature_batch)

        return final_img
