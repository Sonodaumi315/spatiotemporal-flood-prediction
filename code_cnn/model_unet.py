import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Double_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, res = False):
        super(Double_Conv2d, self).__init__()
        self.res = res
        self.leakyrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope = 0.2, inplace=False),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channel),
            )
        if self.res:
            if in_channel != out_channel:
                self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding='same', bias=False)

    def forward(self, inp):
        res = inp
        out = self.conv1(inp)
        out = self.conv2(out)
        if self.res:
            if hasattr(self, "conv3"):
                res = self.conv3(res)
            out += res
            
        out = self.leakyrelu(out)
        return out
    
class Up_Block(nn.Module):
    def __init__(self, in_channel, out_channel, res = False, scale_factor = 2, pad_mode = 0):
        super(Up_Block, self).__init__()
        self.conv = Double_Conv2d(in_channel+out_channel, out_channel, res = res)
        self.up = nn.Upsample(scale_factor = scale_factor, mode = 'bilinear')
        self.pad_mode = pad_mode
        
    def forward(self, inputs2, inputs1):
        
        results2 = self.up(inputs2)
        if self.pad_mode == 0:
            padding = (results2.size()[-1] - inputs1.size()[-1]) // 2
            results1 = F.pad(inputs1, 2 * [padding, padding])
            results = torch.cat([results1, results2], 1)
        elif self.pad_mode == 1:
            padl = (inputs1.size()[-1] - results2.size()[-1]) // 2
            padr = padl+(inputs1.size()[-1] - results2.size()[-1]) % 2
            padu = (inputs1.size()[-2] - results2.size()[-2]) // 2
            padb = padu+(inputs1.size()[-2] - results2.size()[-2]) % 2
            results2 = F.pad(results2, [padl, padr, padu, padb])
            results = torch.cat([inputs1, results2], 1)
        return self.conv(results)

class UNET(nn.Module):
    def __init__(self, in_channel = 6, out_channel = 5, grid_size = 64, base_filters = 64, ndepth = 3, res = False, diff_scale_factor = False, pad_mode = 0):
        super(UNET, self).__init__()
        self.in_channel   = in_channel
        self.out_channel  = out_channel
        self.base_filters = base_filters
        self.ndepth       = ndepth
        self.res          = res
        self.MSELoss      = nn.MSELoss()
        self.convf        = nn.Conv2d(self.base_filters, self.out_channel, kernel_size=1, padding='same')
        self.diff_scale_factor = diff_scale_factor
        self.pad_mode     = pad_mode
        self.norm         = nn.BatchNorm2d(in_channel)
        
        #filters = [16, 32, 64, 128, 256]  
        self.filters = []
        self.grid_size = []

        for i in range(self.ndepth):
            a = self.base_filters * (2**i)
            if a > 512:
                a = 512
            self.filters.append(a)
            self.grid_size.append(grid_size)
            grid_size = grid_size // 2
            if grid_size%2 == 1:
                grid_size -= 1
        
        self.conv0 = Double_Conv2d(self.in_channel, self.filters[0], res = res)
        for i in range(1, self.ndepth):
            setattr(self, "conv%d"%(i), Double_Conv2d(self.filters[i-1], self.filters[i], res = res))
        
        for i in range(self.ndepth):
            if self.grid_size[i]%4 == 2:
                setattr(self, "avgpool%d"%(i), nn.AvgPool2d(kernel_size=4, stride=2))
            else:
                setattr(self, "avgpool%d"%(i), nn.AvgPool2d(kernel_size=2))
        if diff_scale_factor:
            scale_factor = self.grid_size[-1]/grid_size
        else:
            scale_factor = 2
        setattr(self, "upsample%d"%(self.ndepth-1), Up_Block(self.filters[self.ndepth-1], self.filters[self.ndepth-1], res = res, scale_factor = scale_factor, pad_mode = self.pad_mode))
        for i in range(self.ndepth-2, -1, -1):
            if diff_scale_factor:
                scale_factor = self.grid_size[i]/self.grid_size[i+1]
            else:
                scale_factor = 2
            setattr(self, "upsample%d"%(i), Up_Block(self.filters[i+1], self.filters[i], res = res, scale_factor = scale_factor, pad_mode = self.pad_mode))
        
        self.get_info()

    def get_info(self):
        print("---UNET config---")
        print(" in_channel ", self.in_channel)
        print(" out_channel ", self.out_channel)
        print(" filters ", self.filters)
        print(" grid_size ", self.grid_size)
        print(" res ", self.res)
        
    #@autocast()
    def forward(self, inputs):
        inputs = self.norm(inputs)
        skip_x = []
        for i in range(self.ndepth):
            if i == 0:
                x = self.conv0(inputs)
            else:
                x = getattr(self, "conv%d"%(i))(a)
            #print(" rnm ", x.shape)
            skip_x.append(x)
            a = getattr(self, "avgpool%d"%(i))(x)

        for i in range(self.ndepth-1, -1, -1):
            a = getattr(self, "upsample%d"%(i))(a, skip_x[i])
        results = self.convf(a)
        return results

class BaseCNN(nn.Module):
    def __init__(self, base_filters = 64, ndepth = 4, bd = 2, ifr = 1, res = False, grid_size = [64, 64], pad_mode = 0, **args):
        super(BaseCNN, self).__init__()
        
        self.base_filters = base_filters
        self.ndepth       = ndepth
        self.res          = res
        self.bd           = bd
        self.ifr          = ifr
        self.grid_size    = grid_size
        self.pad_mode     = pad_mode
        self.MSELoss      = nn.MSELoss()
        self.model        = UNET(in_channel = 3+4*ifr, out_channel = 3, grid_size = 64+bd*2, base_filters = base_filters, ndepth = ndepth, res = res, diff_scale_factor = False, pad_mode = self.pad_mode)  #h,z,e,u,v
        
        self.get_info()

    def get_info(self):
        print("---BaseCNN config---")
        print(" base filters", self.base_filters)
        print(" ndepth: ", self.ndepth)
        print(" res ", self.res)
        print(" boundary ", self.bd)
        print(" ifr ", self.ifr)
        print(" grid_size ", self.grid_size)
        print(" pad_mode ", self.pad_mode)
        
    #@autocast()
    def forward(self, sx, dx1, dx2 = None, bc = None, bc2 = None, norm_param = None, **args):
        
        bd  = self.bd
        ifr = self.ifr
        gs  = self.grid_size
        sx  = sx.clone()
        dx1 = dx1.clone()
        if dx2 is not None:
            dx2 = dx2.clone()
        
        if norm_param is not None:
            for i in range(3):
                sx[:, i, :, :] = (sx[:, i, :, :] - norm_param[i][0])/norm_param[i][1]
            for i in range(4):
                for j in range(ifr):
                    dx1[:, i+j*4, :, :] = (dx1[:, i+j*4, :, :] - norm_param[3+i][0])/norm_param[3+i][1]
                dx2[:, i, :, :] = (dx2[:, i, :, :] - norm_param[3+i][0])/norm_param[3+i][1]

        inputs  = torch.cat([sx, dx1], dim = 1)
        temp = self.model(inputs)
        if self.pad_mode == 0:
            results = torch.add(dx1[:, :3, bd:gs[0]+bd, bd:gs[1]+bd], temp)
        elif self.pad_mode == 1:
            results = torch.add(dx1[:, :3], temp)[:, :, bd:gs[0]+bd, bd:gs[1]+bd]
        
        if dx2 is not None:
            dx2 = dx2[:, :-1, :, :].view(results.shape)
            mseloss = self.MSELoss(results, dx2)
        else:
            mseloss = None
            
        rt_results = results.clone()
        if norm_param is not None:
            for i in range(3):
                rt_results[:, i, :, :] = rt_results[:, i, :, :]*norm_param[3+i][1] + norm_param[3+i][0]
                
        
        output = {
            "loss": mseloss,
            "results": rt_results
        }
        return output

if __name__ == "__main__":
    sx = torch.randn((4, 3, 101, 74), dtype = torch.float32) 
    dx1 = torch.randn((4, 4, 101, 74), dtype = torch.float32) 
    model = BaseCNN(base_filters = 64, ndepth = 4, bd = 1, ifr = 1, grid_size = [99, 72], pad_mode = 1)
    output = model(sx = sx, dx1 = dx1)