import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
# from models.positional_encoding import PositionalEncoding

class DoubleConv2d(nn.Module):

    def __init__(self, Cin_DConv, Cout_DConv):
        super(DoubleConv2d, self).__init__()

        # side ways double 3x3 2D convolution
        # BLUE ARROWS in the diagram
        self.sideconv = nn.Sequential(
                                nn.Conv2d(Cin_DConv, Cout_DConv, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(Cout_DConv),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(Cout_DConv, Cout_DConv, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(Cout_DConv),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.sideconv(x)

class UNet(nn.Module):

    def __init__(self, Cin_UNet, features, dropout, image_length):
        super(UNet, self).__init__()

        # initializing ModuleLists to hold array of steps of nn.Modules
        # for both down path i.e. encoder and up-part i.e. decoder
        self.down_side_net = nn.ModuleList()
        self.up_net = nn.ModuleList()
        self.up_side_net = nn.ModuleList()

        # RED ARROWS in the diagram
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part or encoder part
        for feat in features:
            self.down_side_net.append(DoubleConv2d(Cin_UNet, feat))
            Cin_UNet = feat

        # Bottleneck or the last layer
        self.bottleconv = DoubleConv2d(features[-1], features[-1]*2)

        # Up part or decoder part
        for feat in reversed(features):
            # GREEN ARROWS in the diagram
            self.up_net.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.up_side_net.append(DoubleConv2d(feat*2, feat))

        # final layer of the U-Net        
        self.final_conv = nn.Conv2d(features[0], Cin_UNet, kernel_size=1, stride=1)
        
        # positional encoding
        # self.pe = PositionalEncoding(Cin_UNet, dropout, image_length)

    def forward(self, x):

        # run down steps and save skip_connections
        skip_connections = []
        for down in self.down_side_net:
            # x channels from feature[i] to feature[i+1]
            x = down(x)
            # save the skip_connections
            skip_connections.append(x)
            # maxpool i.e. step down in diagram
            x = self.maxpool(x)


        # till this step: x size is [B, 512, W, H]
        # run Bottleneck convolution layer: x size will be [B, 1024, W', H']
        x = self.bottleconv(x)

        # to make coding easier, reverse the skip_connections
        skip_connections = skip_connections[::-1]
        # run the up steps
        for idx in range(len(self.up_net)):
            # Up step
            x = self.up_net[idx](x)  # [B, 512, w, h]
            if skip_connections[idx].shape != x.shape:
                x = ttf.resize(x, size=skip_connections[idx].shape[2:])

            # concat skip_connections and x
            x = torch.cat((skip_connections[idx], x), dim=1)
            # run 3x3 conv2d layer side ways
            x = self.up_side_net[idx](x)

        # final convolution layer
        x  = self.final_conv(x)

        # saving some random tensors to visualize segmented image
        torch.save(x, "segmented_images/tnsr.pt")

        exit()
        # positional encoding
        x = torch.flatten(x, 2, -1) # (B, features[0], length)
        x = x.permute(2, 0, 1)  # (length, B, features[0])
        x += self.pe(x) # (length, B, features[0])

        return x


unet = UNet(
    Cin_UNet=4, 
    features=[64,128,256,512],
    dropout=0.1,
    image_length=320*480,
)

x=torch.load("/groups/claytonm/gauravs_data/clevrmath_data/data/image_tensors/10.pt")
unet(x.unsqueeze(0))