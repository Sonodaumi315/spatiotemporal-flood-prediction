import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import cv2
from PIL import Image
import time


WD_color = LinearSegmentedColormap.from_list('', ['white', 'MediumBlue'])
Q_color = LinearSegmentedColormap.from_list('', ['red', 'white', 'green'])
V_color = LinearSegmentedColormap.from_list('', ['red', 'white', 'green'])

diff_color = LinearSegmentedColormap.from_list('', ['red', 'white', 'green'])
diff_color = LinearSegmentedColormap.from_list('', ['aqua', 'white', 'purple'])
diff_color_positive = LinearSegmentedColormap.from_list('', ['white', 'green'])
diff_color_negative = LinearSegmentedColormap.from_list('', ['red', 'white'])
FAT_color = LinearSegmentedColormap.from_list('', ['MediumBlue', 'white'])

def find_connected_block(qs, output, dem, lastmap, q, dx2, steps = [1, 1, 1], print_details = False):
    lastmap = lastmap.copy()
    device = output.device
    q = abs(q).cpu().numpy()
    delta_wd = q*36
    output = output.cpu().numpy()
    dx2 = dx2.cpu().numpy()
    dem = dem.cpu().numpy()
    eps = 1e-4
    nx = dem.shape[-2]
    ny = dem.shape[-1]

    if steps[0] == 1:
        mask = output[0, 0, :, :] <= eps
        for i in range(3):
            output[0, i][mask] = 0
    
    mask = np.zeros((nx, ny), dtype = np.bool_)
    corrected_wd = np.zeros((nx, ny), dtype = np.float32)

    queue = []
    for qq in qs:
        sx, sy = qq
        if output[0, 0, sx, sy] > 0:
            queue.append(qq)
            mask[sx][sy] = True 
            corrected_wd[sx][sy] = 1e9
    
    if steps[1] == 1:
        st = 0
        sum_wd2 = 0
        while st < len(queue):
            ox, oy = queue[st]
            st += 1
            q2 = []
            if ox-1 >= 0:
                q2.append([ox-1, oy])
            if ox+1 < 64:
                q2.append([ox+1, oy])
            if oy-1 >= 0:
                q2.append([ox, oy-1])
            if oy+1 < 64:
                q2.append([ox, oy+1])
            for qq in q2:
                x, y = qq
                if (output[0, 0, x, y] > eps) and (mask[x][y] == False) and (dem[x][y] <= dem[ox][oy] + output[0, 0, ox, oy]):
                    mask[x][y] = True
                    queue.append(qq)

            sum_wd2 += output[0, 0, ox, oy]
        
        lastmap[mask == False] = 0
        mask = np.repeat(np.expand_dims(mask, 0), 3, 0)
        output[0][mask == False] = 0

    if steps[2] == 1:
        mask2 = output[0, 0] > 0
        sum_wd2 = np.sum(output[0, 0])
        sum_wd = np.sum(lastmap)
        dwd = output[0, 0] - lastmap
        dwd *= (sum_wd2 - (sum_wd + delta_wd))/(sum_wd2 - sum_wd)
        rt = (output[0][0][mask2] - dwd[mask2])/output[0][0][mask2]
        for i in range(3):
            output[0][i][mask2] *= rt

    return torch.tensor(output, dtype = torch.float32).to(device)

def find_boundary(x, y, l, inv = False):
    if inv == False:
        if x == 0:
            x -= 1
        elif x == l-1:
            x += 1
        elif y == 0:
            y -= 1
        elif y == l-1:
            y += 1
    else:
        if x == -1:
            x += 1
        elif x == l:
            x -= 1
        elif y == -1:
            y += 1
        elif y == l:
            y -= 1
        
    return x, y

def get_num(v):
    if v<10:
        a="00%d"%(v)
    elif v<100:
        a="0%d"%(v)
    else:
        a="%d"%(v)
    return a

def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def div(a, b):
    if b == 0:
        return 0
    return a/b

def generate_mp4(img_list, video_size, save_path, fps = 1):
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(save_path, fourcc, fps, video_size)
    for img in img_list:
        video.write(img)
           
def add_breach_location(ax, BC, ng):
    if BC is not None:
        bx, by = find_boundary(BC[0], BC[1], ng, True)
        ax.scatter(by, bx, c='r', marker='*')

def add_colorbar(ax, cmap, maxv, minv, difference):
    if ax is None: 
        ax = plt.gca()
    
    if maxv == 0 or minv == 0:
        num_level = 6
    else:
        num_level = 7
    num_level = 5
    ticks_interval = np.linspace(minv, maxv, num_level, endpoint=True)
    
    if (difference is True) and (minv < 0) and (maxv > 0):
        SM = plt.cm.ScalarMappable(norm = TwoSlopeNorm(vmin = minv , vcenter = 0, vmax = maxv), cmap = cmap)
    else:
        SM = plt.cm.ScalarMappable(norm = plt.Normalize(vmin = minv, vmax = maxv), cmap = cmap)
    plt.colorbar(SM, ticks = np.sign(ticks_interval)*np.floor((np.abs(ticks_interval)*100)+1e-6)/100, fraction = 0.05, shrink = 0.9, ax = ax)
        
def plot_map(map, ax = None, norm_param = None, var = "WD", maxv = None, minv = None, difference = False, colorbar = False):
    if map.shape[0] == 1:
        map = map[0]
    
    if norm_param is not None:
        map = map/norm_param[var][1] + norm_param[var][0]
    
    if ax is None:
        ax = plt.gca()
    
    if map.shape[0] > 64:
        r'''
        ax.set_xticks([0,40,80,120]) 
        ax.set_xticklabels([0,4,8,12])
        ax.set_yticks([0,40,80,120]) 
        ax.set_yticklabels([0,4,8,12])
        '''
        ax.set_xticks([0,20,40,60]) 
        ax.set_xticklabels([0,2,4,6])
        ax.set_yticks([0,20,40,60,80]) 
        ax.set_yticklabels([0,2,4,6,8])
    else:
        ax.set_xticks([0,20,40,60]) 
        ax.set_xticklabels([0,2,4,6])
        ax.set_yticks([0,20,40,60]) 
        ax.set_yticklabels([0,2,4,6])
            
    X = []
    Y = []
    V = []
    for j in range(map.shape[1]):
        for i in range(map.shape[0]):
            X.append(i)
            Y.append(j)
            V.append(map[i][j])
    X = np.asarray(X)
    Y = np.asarray(Y)
    V = np.asarray(V)
    
    if maxv is None:
        maxv = max(map)
    if minv is None:
        minv = min(map)
        
    if difference:
        if maxv < 0:
            maxv = 0
            cmap = diff_color_negative
        elif minv > 0:
            minv = 0
            cmap = diff_color_positive
        else:
            cmap = diff_color
    else:
        if var == "DEM":
            cmap = "terrain"
        elif var == "WD":
            cmap = WD_color
        elif var == "VX" or var == "VY":
            cmap = V_color
        elif var == "QX" or var == "QY":
            cmap = Q_color
            
    #ax.tricontourf(X, Y, V, levels = 48, cmap = cmap, interpolation = None, origin = 'lower')
    #ax.tricontourf(X, Y, V, cmap = cmap)
    
    if colorbar:
        add_colorbar(ax = ax, cmap = cmap, maxv = maxv, minv = minv, difference = difference)
    
    ax.imshow(map, vmin = minv, vmax = maxv, cmap = cmap, origin='lower')
    
def print_frame(DEM, BC, frame, timestep, real_flood, predicted_flood, model_list, var_list, norm_param = None, maxv_list = None, minv_list = None,
                save_path = None, return_pic = False):
    fig, axs = plt.subplots(len(var_list), 2+2*len(model_list), figsize=(2 + 3*(2+2*len(model_list)), 3*len(var_list)), facecolor='white', 
                            gridspec_kw={'width_ratios': [1]*(2+2*len(model_list))}, constrained_layout = True)
        
    if maxv_list is None:
        maxv_list=[
            #[2.5, 2],
            [5, 2],
            [0.3, 0.3],
            [0.3, 0.3]
        ]
    
    if minv_list is None:
        r'''
        minv_list=[
            [0, -2],
            [0, -0.6]
        ]'''
        minv_list=[
            [0, -2],
            [-0.3, -0.3],
            [-0.3, -0.3]
        ]
    plot_map(map = DEM, ax = axs[0,0], norm_param = norm_param, var = "DEM", maxv = np.max(DEM), minv = np.min(DEM), colorbar = True)
    add_breach_location(ax = axs[0,0], BC = BC, ng = DEM.shape[-1])
    axs[0,0].set_xlabel("y distance (km)")
    axs[0,0].set_ylabel("x distance (km)")
    axs[0,0].set_title('DEM(m)')
    
    axs[0,1].set_title('Ground-truth')
    axs[0,1].set_xlabel("time = %.1lf h"%(timestep/2))
    
    num_model = len(model_list)
    for i in range(num_model):
        model_name = model_list[i].replace("<class \'model_phycnn.", "")
        model_name = model_name.replace("<class \'model_phyunet.", "")
        model_name = model_name.replace("\'>", "")
        axs[0,2+i].set_title("%s Prediction"%(model_name))
        axs[0,2+num_model+i].set_title("%s Difference"%(model_name))
        
        
    for j in range(len(var_list)):
        var = var_list[j]
        if var == "WD":
            unit = "m"
        elif var == "VX" or var == "VY":
            unit = "m/s"
        elif var == "QX" or var == "QY":
            unit = "$m^2$/s"
        ylabel = "%s(%s)"%(var, unit)
        axs[j,1].set_ylabel(ylabel)
        
        
        plot_map(map = real_flood[var][frame], ax = axs[j,1], norm_param = norm_param, var = var, maxv = maxv_list[j][0], minv = minv_list[j][0])
        
        for i in range(num_model):
            if i == num_model-1:
                colorbar = True
            else:
                colorbar = False
                
            plot_map(map = predicted_flood[0][var][frame], ax = axs[j,2+i], norm_param = norm_param, 
                     var = var, maxv = maxv_list[j][0], minv = minv_list[j][0], colorbar = colorbar)
                        
            diffs = predicted_flood[0][var][frame] - real_flood[var][frame]
            plot_map(map = diffs, ax = axs[j,2+num_model+i], norm_param = norm_param, 
                     var = var, maxv = maxv_list[j][1], minv = minv_list[j][1], difference = True, colorbar = colorbar)
            
            axs[j,2+num_model+i].set_xlabel("RMSE(%s): %.3lf"%(unit, np.sqrt(np.mean((diffs)**2))))
        

    axs[1,0].axis('off')
    axs[2,0].axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    if return_pic:
        plt.savefig("temp.png")
        img = cv2.imread("temp.png")
        img_size = Image.open("temp.png").size
        return img, img_size
