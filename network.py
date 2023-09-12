import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Iterable,Union

class UNet(nn.Module):
    def  __init__(self, 
                in_channels:int,
                layer_channels:Iterable[int]=[32,64,128],
                out_channels:Union[None,Iterable[int]]=None,
                downsample_scale:int=3,
                kernel_size:int=5,
                n_convs_per_layer = 2, 
                dropout=0.3,
                batch_norm=True
                ):
        super(UNet,self).__init__()
        """
        A small U-net model, with skip-attention, and downsampling and upsampling
        :param layer depths: the number of channels at each layer, passed to lower model, 
        :param out_depths: Default None, the channel output returned with forward model. 
	                By default this will be the same as the input size, but can be specified. 
		            If specified, this hould be the same length as layer_depths
		:param downsample_scale: the scale by which to downsample, by default this is 2x, but can be increased
		:param kernel_size: the size of kernels to be used. 3 by default; typically an odd number
        """
        if out_channels  is None:
            out_channels = [in_channels] + layer_channels[:-1]
        assert(len(layer_channels) == len(out_channels))
        assert(len(layer_channels) >0)

        # two cases... 
        if len(layer_channels) == 1: 
            # downsample then immediately upsample
            self.pre_convs = nn.ModuleList()
            self.pre_batch_norms = nn.ModuleList()
            for i in range(n_convs_per_layer):
                in_chans = in_channels if i == 0 else layer_channels[0]
                out_chans = out_channels[0] if i == n_convs_per_layer-1 else layer_channels[0]
                print(in_chans,out_chans)
                self.pre_convs.append(nn.Conv2d(in_chans,out_chans,kernel_size=kernel_size,padding="same"))
                self.pre_batch_norms.append(nn.BatchNorm2d(out_chans) if  batch_norm and i != n_convs_per_layer-1 else nn.Identity())
            self.sublayer = None 
        else: 
            # in block 
            self.pre_convs = nn.ModuleList()
            self.pre_batch_norms = nn.ModuleList()
            for i in range(n_convs_per_layer):
                in_chans = in_channels if i == 0 else layer_channels[0]
                out_chans = layer_channels[0] 
                print(in_chans,out_chans)

                self.pre_convs.append(nn.Conv2d(in_chans,out_chans,kernel_size=kernel_size,padding="same"))
                self.pre_batch_norms.append(nn.BatchNorm2d(out_chans) if batch_norm else nn.Identity())

            # downsample, dropout, sublayer
            self.downsample = nn.MaxPool2d(kernel_size=downsample_scale,stride=downsample_scale)
            # down 
            self.dropout= nn.Dropout(p=dropout)
            self.sublayer = UNet(in_channels=layer_channels[0],
                                 layer_channels=layer_channels[1:],
                                 out_channels=out_channels[1:],
                                 downsample_scale=downsample_scale,
                                 kernel_size=kernel_size,
                                 dropout=dropout)
            # up 
            self.upsample = nn.Upsample(scale_factor=downsample_scale)
            self.post_convs = nn.ModuleList()
            self.post_batch_norms = nn.ModuleList()
            for i in range(n_convs_per_layer):
                in_chans = out_channels[1] + layer_channels[0]  if i == 0 else layer_channels[0]
                out_chans = out_channels[0] if i == n_convs_per_layer-1 else layer_channels[0]
                print(in_chans,out_chans,out_channels[1])

                self.post_convs.append(nn.Conv2d(in_chans,out_chans,kernel_size=kernel_size,padding="same"))
                self.post_batch_norms.append(nn.BatchNorm2d(out_chans) 
                                                if batch_norm and i != n_convs_per_layer-1 
                                                else nn.Identity())

    def forward(self, input):
        x = input
        if self.sublayer is None: 
            for i,(bn, conv) in enumerate( zip(self.pre_batch_norms,self.pre_convs)):
                if i == len(self.pre_convs)-1:
                    # last layer, dont batch norm or do activation 
                    x = conv(x)
                else:   
                    x = F.relu(bn(conv(x)))
        else: 
            # run in block
            for bn, conv in zip(self.pre_batch_norms,self.pre_convs):
                x = F.relu(bn(conv(x)))
            # cache input x, target shape for upsampling
            x_pre = x 
            upsample_shape = x_pre.shape[-2:]
            # then downsample 
            x = self.downsample(x)
            x = self.dropout(x) 
            # run sub layer, relu output 
            x_post = F.relu(self.sublayer(x))

            # upsample and add output from  'down' block, and add dropout
            x_post = F.upsample(x_post,upsample_shape)
            x = torch.concat((x_pre,x_post),dim=-3) 
            x = self.dropout(x)

            # up block. Additional logic is to avoid doing relu on last element 
            for i, (bn, conv) in enumerate(zip(self.post_batch_norms,self.post_convs)):
                if i == len(self.post_convs)-1:
                     # last layer 
                    x = conv(x)
                else: 
                    x = F.relu(bn(conv(x)))
        return x

class DownsampleLayer(nn.Module): 
    def __init__(self,in_size, out_size, kernel_size,stride,do_batch_norm:bool=True,dropout:Union[float,None]=None ):
        super(DownsampleLayer,self).__init__()
        pad_size = (kernel_size -1)//2
        self.conv= nn.Conv2d(in_channels=in_size,out_channels=out_size,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.batch_norm = nn.BatchNorm2d(out_size) if do_batch_norm else None 
        self.dropout = None if dropout is None or dropout == 0 else nn.Dropout(p=dropout)
    
    def forward(self,input):
        x = input 
        if self.dropout is not None: 
            x = self.dropout(x)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return x 

class UpsampleLayer(nn.Module): 
    def __init__(self,in_size, out_size, kernel_size,stride,do_batch_norm:bool=True,dropout:Union[float,None]=None ):
        super(UpsampleLayer,self).__init__()
        pad_size = (kernel_size -1)//2 # same ish padding 
        self.conv= nn.ConvTranspose2d(in_channels=in_size,out_channels=out_size,kernel_size=kernel_size,stride=stride,padding=pad_size)
        self.batch_norm = nn.BatchNorm2d(out_size) if do_batch_norm else None 
        self.dropout = None if dropout is None else nn.Dropout(p=dropout)
    
    def forward(self,input):
        x = input 
        if self.dropout is not None: 
            x = self.dropout(x)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return x 



class Small(nn.Module):
    def __init__(self, dropout = 0.3,kernel_size=4,downscale_by=2, batch_norm = False,activation = nn.ReLU ): 
        super(Small,self).__init__()
        self.activation = activation()
        # downsample three x [32,64,128]
        self.down1 = DownsampleLayer(1,32,kernel_size,downscale_by,batch_norm,False) # dont dropout for 1 channel input, as tht would remove all info 
        self.down2 = DownsampleLayer(32,64,kernel_size,downscale_by,batch_norm,dropout)
        self.down3 = DownsampleLayer(64,128,kernel_size,downscale_by,batch_norm,dropout)
        # same-sample 2x [128,128]
        # use same layers by just not using stride  
        self.same1 = DownsampleLayer(128,128,kernel_size,1,batch_norm,dropout)
        self.same2 = DownsampleLayer(128,128,kernel_size,1,batch_norm,dropout)
        self.same3 = DownsampleLayer(128,128,kernel_size,1,batch_norm,dropout)
        self.same4 = DownsampleLayer(128,128,kernel_size,1,batch_norm,dropout)
        self.same5 = DownsampleLayer(128,128,kernel_size,1,batch_norm,dropout)

        # up-sample 3x [64,32, 2 ]
        self.up1 = UpsampleLayer(128,64,kernel_size,downscale_by,batch_norm,dropout)
        self.up2 = UpsampleLayer(64,32,kernel_size,downscale_by,batch_norm,dropout)
        self.up3 = UpsampleLayer(32,2,kernel_size,downscale_by,False,dropout) # batch norm not done on last level 
    
    def forward(self,input):
        in_size = input.shape[-2:]
        # down 
        xd1 = self.activation(self.down1(input))
        xd2 = self.activation(self.down2(xd1))
        xd3 = self.activation(self.down3(xd2))
        # same 
        x = self.activation(self.same1(xd3))
        x = self.activation(self.same2(x))
        x = self.activation(self.same3(x))
        x = self.activation(self.same4(x))
        x = self.activation(self.same5(x))

        # up  
        x = self.activation(self.up1(x))
        x = self.activation(self.up2(x))
        h_out = self.up3(x)
        # rectify size 
        return F.upsample(h_out,in_size) 

        # return h_out 
        
