import argparse
import json
import os
from tqdm.auto import tqdm
import numpy as np
import math
import torch
import shutil
import openpyxl
from model_unet import BaseCNN
import torch.distributed as dist


from utils import reduce_sum, div, print_frame, generate_mp4, find_connected_block
from dataset import FloodDatasetCNN, make_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--augs', type=bool, default=False,
                help='data augmentation')
parser.add_argument('--eval_batch_size', type=int, default=4,
                help='eval batch size')
parser.add_argument('--print_loss', type=bool, default=False,
                help='print loss')

parser.add_argument('--temporal_resolution', type=int, default=2,
                help='temporal resolution')
parser.add_argument('--simulation_steps', type=int, default=120,
                help='simulation steps')
parser.add_argument('--input_frames', type=int, default=2,
                help='input frames')

parser.add_argument('--train_data_seed', type=list, default=None,
                help='train data seed')
parser.add_argument('--valid_data_seed', type=list, default=None,
                help='valid data seed')
parser.add_argument('--norm_param', type=str, default=None,
                help='normalization parameters')
parser.add_argument('--boundary', type=int, default=1,
                help='boundary')
parser.add_argument('--scene', type=int, default=0,
                help='0:synthetic 1:bentivogilio 2:tous dam')

parser.add_argument('--base_filters', type=int, default=64,
                help='base_filters')
 
def make_mp4(DEM, BC, real_flood, predicted_flood, model_list, var_list, st, tr, save_path):
    img_list = []
    num = len(real_flood[var_list[0]])
    ts = st
    for i in range(num):
        img, img_size = print_frame(DEM = DEM, BC = BC, frame = i, timestep = ts, real_flood = real_flood, predicted_flood = predicted_flood, model_list = model_list, var_list = var_list, return_pic = True)
        img_list.append(img)
        print("###make_mp4  %d/%d"%(i, num))
        ts += tr
    generate_mp4(img_list = img_list, video_size = img_size, save_path = save_path, fps = 3)
   
def cal(prefix, tot_loss, flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, print_loss = True):
    
    flMSE = div(flSSE, flpix)
    csi05 = div(l_H_005, (l_H_005+l_M_005+l_FP_005))
    csi30 = div(l_H_030, (l_H_030+l_M_030+l_FP_030))
    p05   = div(l_H_005, (l_H_005+l_FP_005))
    p30   = div(l_H_030, (l_H_030+l_FP_030))
    r05   = div(l_H_005, (l_H_005+l_M_005))
    r30   = div(l_H_030, (l_H_030+l_M_030))
    f05   = div(2*p05*r05, (p05+r05))
    f30   = div(2*p30*r30, (p30+r30))
    
    if print_loss:
        print("   tot_loss:%.6lf"%(tot_loss))
    print("   flMSE:%.6lf=%.6lf/%d    %s-RMSE:%.6lf"%(flMSE, flSSE, flpix, prefix, math.sqrt(flMSE)))
    print("   %s-030cases %d %d %d    acc_005:%.6lf acc_030:%.6lf csi_005:%.6lf csi_030:%.6lf  fs_005:%.6lf fs_030:%.6lf"%(prefix, l_H_030, l_M_030, l_FP_030, div(l_ACC_005, l_TOTAL), div(l_ACC_030, l_TOTAL), csi05, csi30, f05, f30))
    return flMSE

def eval_batch(preds, label, flSSE = 0, flpix = 0, l_ACC_005 = 0, l_ACC_030 = 0, l_H_005 = 0, l_H_030 = 0, l_M_005 = 0, l_M_030 = 0, l_FP_005 = 0, l_FP_030 = 0, l_TOTAL = 0, print_detail = None):
    sp = preds.shape
    l_TOTAL += sp[0]*sp[-1]*sp[-2]
    
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.detach().cpu().numpy()
    
    
    b = preds[:, 0, :, :]
    a = label[:, 0, :, :]
    
    fl_obs  = (a > 0.05)
    fl_pred = (b > 0.05)
    
    flselect  = (a > -1e9)
    flpix    = flpix + np.sum(flselect)
    flSSE    = flSSE + np.sum(np.power(a[flselect]-b[flselect], 2))

    #
    H  = np.logical_and(fl_obs, fl_pred)
    M  = np.logical_and(fl_obs, np.logical_not(fl_pred))
    FP = np.logical_and(np.logical_not(fl_obs), fl_pred)
    TN = np.logical_and(np.logical_not(fl_obs), np.logical_not(fl_pred))
    #
    H  = np.sum(H)
    M  = np.sum(M)
    FP = np.sum(FP)
    TN = np.sum(TN)
    #
    l_ACC_005 = l_ACC_005 + (H+TN)
    l_H_005   = l_H_005   + H
    l_M_005   = l_M_005   + M
    l_FP_005  = l_FP_005  + FP
    
    #
    fl_obs  = (a > 0.3)
    fl_pred = (b > 0.3)
        
    #
    H  = np.logical_and(fl_obs, fl_pred)
    M  = np.logical_and(fl_obs, np.logical_not(fl_pred))
    FP = np.logical_and(np.logical_not(fl_obs), fl_pred)
    TN = np.logical_and(np.logical_not(fl_obs), np.logical_not(fl_pred))
    #
    H  = np.sum(H)
    M  = np.sum(M)
    FP = np.sum(FP)
    TN = np.sum(TN)
    #
    l_ACC_030 = l_ACC_030 + (H+TN)
    l_H_030   = l_H_030   + H
    l_M_030   = l_M_030   + M
    l_FP_030  = l_FP_030  + FP
    
    torch.cuda.empty_cache()
    
    if print_detail is not None:
        cal(print_detail, -1, flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, False)
    
    return flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL
        
def evaluate(prefix, model, norm_param, val_generator, local_rank = -1, return_loss = False, print_detail = False):
    
    l_ACC_005 = 0; l_ACC_030 = 0
    l_H_005   = 0; l_H_030   = 0
    l_M_005   = 0; l_M_030   = 0
    l_FP_005  = 0; l_FP_030  = 0
    l_TOTAL   = 0
    flpix = 0
    flSSE = 0
    
    cnt = 0
    tot_loss = 0
    tot_sample = 0
    model.eval()
    if local_rank != -1:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")
    
    for batch in tqdm(val_generator):
        
        with torch.no_grad():
            sx = batch[0].to(device, non_blocking=True)
            dx1 = batch[1].to(device, non_blocking=True)
            dx2 = batch[2].to(device, non_blocking=True)
            bc  = batch[3].to(device, non_blocking=True)
            bc2 = batch[4].to(device, non_blocking=True)
            output = model(sx = sx, dx1 = dx1, dx2 = dx2, bc = bc, bc2 = bc2, norm_param = norm_param)
            tot_loss += output["loss"]
        
        if print_detail:
            print_detail_prefix = "valid_sample_%d "%(cnt)
        else:
            print_detail_prefix = None
            
        eval_batch(output["results"], batch[2], print_detail = print_detail_prefix)
        flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = eval_batch(output["results"], batch[2],
        flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL)
        
        cnt += 1
        tot_sample += batch[0].shape[0]

            
    if local_rank != -1:
        l_ACC_005 = torch.tensor(l_ACC_005, dtype=float).to(device); l_ACC_030 = torch.tensor(l_ACC_030, dtype=float).to(device)
        l_H_005   = torch.tensor(l_H_005, dtype=float).to(device);   l_H_030   = torch.tensor(l_H_030, dtype=float).to(device)
        l_M_005   = torch.tensor(l_M_005, dtype=float).to(device);   l_M_030   = torch.tensor(l_M_030, dtype=float).to(device)
        l_FP_005  = torch.tensor(l_FP_005, dtype=float).to(device);  l_FP_030  = torch.tensor(l_FP_030, dtype=float).to(device)
        l_TOTAL   = torch.tensor(l_TOTAL, dtype=float).to(device)
        flSSE     = torch.tensor(flSSE, dtype=float).to(device)
        flpix     = torch.tensor(flpix, dtype=float).to(device)
        tot_loss  = torch.tensor(tot_loss, dtype=float).to(device)
        cnt       = torch.tensor(cnt, dtype=float).to(device)
        tot_sample= torch.tensor(tot_sample, dtype=int).to(device)
        l_ACC_005 = reduce_sum(l_ACC_005); l_ACC_030 = reduce_sum(l_ACC_030)
        l_H_005   = reduce_sum(l_H_005);   l_H_030   = reduce_sum(l_H_030)
        l_M_005   = reduce_sum(l_M_005);   l_M_030   = reduce_sum(l_M_030)
        l_FP_005  = reduce_sum(l_FP_005);  l_FP_030  = reduce_sum(l_FP_030)
        l_TOTAL   = reduce_sum(l_TOTAL)
        flSSE     = reduce_sum(flSSE)
        flpix     = reduce_sum(flpix)
        tot_loss  = reduce_sum(tot_loss)
        cnt       = reduce_sum(cnt)
        tot_sample= reduce_sum(tot_sample)
        dist.barrier()
        
    tot_loss /= cnt
    tot_loss = float(tot_loss.detach().cpu().numpy())
    
    if local_rank <= 0:
        print("    tot sample for %s: %d"%(prefix, tot_sample))
        cal(prefix, tot_loss, flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, True)

    if return_loss:
        return tot_loss
    return tot_loss, flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL
    
def eval_valid(args, model_test_path = None, model_class = BaseCNN, results_path = None, device = torch.device("cpu"), overwrite = False):
    
    print("##### start eval valid")
    print(" results_path: %s"%(results_path))
    if (results_path is not None) and os.path.exists(results_path) and (overwrite is False):
        with open(results_path, "r", encoding = "utf-8") as fout:
            results = json.load(fout)
        for i in results:
            print("    %s: %.3lf "%(i, results[i]))
        return 
    
    valid_data_seed = args.valid_data_seed
    simulation_steps = args.simulation_steps
    input_frames = args.input_frames
    temporal_resolution = args.temporal_resolution
    if args.norm_param is None:
        norm_param = None
    else:
        norm_param = np.loadtxt(args.norm_param)

    scene = 0
    pad_mode = 0
    grid_size = [64, 64]
    if "od" in model_test_path.replace("model", "#"):
        scene = 1
        valid_data_seed = range(61, 81)
        args.augs = False
    elif "tous" in model_test_path.replace("model", "#"):
        scene = 2
        valid_data_seed = [9]
        args.augs = False
        pad_mode = 1
        grid_size = [97, 70]
        
        
    print(" model test path: %s"%(model_test_path))
    model = model_class(base_filters = args.base_filters, bd = args.boundary, ifr = args.input_frames, 
                        pad_mode = pad_mode,
                        grid_size = grid_size)
    model.load_state_dict(torch.load(model_test_path))

    print("device: %s"%(device))
    model.to(device)
    
    
    print(" normalization parameters::%s\n "%(args.norm_param), norm_param)
    
        
    valid_dataset = "../data_cnn/valid_%d-%d"%(valid_data_seed[0], valid_data_seed[-1])
    if args.augs is True:
        valid_dataset += "_augs"
    
    valid_dataset, valid_steps = make_dataset(save_path = valid_dataset, seed = valid_data_seed, simulation_steps = simulation_steps, input_frames = input_frames, temporal_resolution = temporal_resolution, 
                                              boundary = args.boundary, shuffle = False, batch_size = args.eval_batch_size, augs = args.augs, scene = scene)
    
    tot_loss, flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = evaluate(prefix = "valid", norm_param = norm_param, model = model, val_generator = valid_dataset, print_detail = True)
    
    if results_path is not None:
        results = {
            "RMSE": math.sqrt(div(flSSE, flpix)), 
            "ratio of flooding area": div(l_H_005+l_FP_005, l_H_005+l_M_005), 
            "AvgCSI": (div(l_H_005, (l_H_005+l_M_005+l_FP_005)) + div(l_H_030, (l_H_030+l_M_030+l_FP_030)))/2, 
            "CSI_0.05": div(l_H_005, (l_H_005+l_M_005+l_FP_005)), 
            "CSI_0.30": div(l_H_030, (l_H_030+l_M_030+l_FP_030)),
            "Loss": tot_loss,
            "ACC_0.05": div(l_ACC_005, l_TOTAL),
            "ACC_0.30": div(l_ACC_030, l_TOTAL)
            }
        with open(results_path, "w", encoding = "utf-8") as fout:
            json.dump(results, fout, indent = 4, ensure_ascii = False)
        shutil.copyfile("eval.log", results_path.replace(".json", ".log"))

def eval_multistep_prediction(args, model_test_path, model_class, test_samples, results_path, use_correction = False, overwrite = False, model = None, device = torch.device("cpu"), make_video = False):
        
    print("##### start eval multistep prediction")
    if use_correction > 1:
        results_path += "_corrected-%d"%(use_correction)
    elif use_correction == True:
        results_path += "_corrected"
    print(" results_path: %s"%(results_path))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    

    scene = 0
    pad_mode = 0
    grid_size = [64, 64]
    if "od" in model_test_path.replace("model", "#"):
        scene = 1
    elif "tous" in model_test_path.replace("model", "#"):
        scene = 2
        pad_mode = 1
        grid_size = [97, 70]
    nx, ny = grid_size
        
    print(" model test path: %s"%(model_test_path))
    if model is None:
        model = model_class(base_filters = args.base_filters, bd = args.boundary, ifr = args.input_frames, 
                        pad_mode = pad_mode,
                        grid_size = grid_size)
        model.load_state_dict(torch.load(model_test_path))        

    if args.norm_param is None:
        norm_param = None
    else:
        norm_param = np.loadtxt(args.norm_param)
    print(" normalization parameters: ", norm_param)
    
    print("device: %s"%(device))
    model.to(device)
    model.eval()
    #var_name = ["h", "qx", "qy", "vx", "vy"]
    
    
    print("----test samples")
    cnt = 0
    for sample in test_samples:
        print("  sample-%d"%(cnt))
        cnt += 1
        for it in sample:
            print("     %s:  %d"%(it, sample[it]))
        tr = sample["temporal_resolution"]
        st = sample["start_timestep"]
        et = sample["end_timestep"]
        #ifr= sample["input_frames"]
        ifr= args.input_frames
        seed = sample["seed"]
        augs = sample["augs"]
        
        bd = args.boundary
        
        var_dic = {
            "WD": 0,
            "QX": 1,
            "QY": 2,
        }
        
        if ( (et-st)%tr > 0 ):
            print("    invalid test sample")
            continue
        ss = "sd-%d tr-%d ifr-%d st-%d et-%d"%(seed, tr, ifr, st, et)
        if augs > 0:
            ss += " augs-%d"%(augs)
        if os.path.exists("%s/%s"%(results_path, ss)) and (overwrite is False):
            print("      %s exists"%(ss))
            continue
        
        results = []
        real_flood = {}
        predicted_flood = {}
        for var in var_dic:
            real_flood[var] = []
            predicted_flood[var] = []
        
        dataset = FloodDatasetCNN(seed = [seed], simulation_steps = et, input_frames = ifr, temporal_resolution = tr, augs = True, bd = bd, scene = scene)
        num = 1 + (et-(ifr-1)*tr - st - 1) // tr
        flood_map = np.zeros((3, num, nx*ny), dtype = np.float32)
        tot = 0
        
        if scene == 0:
            DEM = np.loadtxt("../data/DEM/%d.txt"%(seed))
            DEM -= np.min(DEM)
            DEM = DEM[1:65, 1:65]
            BC  = np.loadtxt("../data/BC/%d.txt"%(seed))
        elif scene == 1:
            DEM = np.loadtxt("../data/Bentivoglio2023/raw_datasets/DEM/DEM_%d.txt"%(seed))
            DEM -= np.min(DEM)
            BC  = None
        elif scene ==2:
            DEM = np.loadtxt("../data_tous/dem/%d.txt"%(seed))
            DEM = np.flip(DEM, 0)
            DEM = DEM[2:99, 2:72]
            BC = [96, 12, 0, 0]
            
            
        if not os.path.exists("%s/%s"%(results_path, ss)):
            os.makedirs("%s/%s"%(results_path, ss))
            
        with torch.no_grad():
            index = st*8 + augs
            sx  = torch.tensor([dataset[index][0]]).to(device, non_blocking=True)
            dx1 = torch.tensor([dataset[index][1]]).to(device, non_blocking=True)
            bc  = torch.tensor([dataset[index][3]]).to(device, non_blocking=True)
            bc2 = torch.tensor([dataset[index][4]]).to(device, non_blocking=True)
            bx, by = int(bc[0][0]+bd), int(bc[0][1]+bd)
            bx2, by2 = int(bc2[0][0]+bd), int(bc2[0][1]+bd)
            dbx = bx2-bx
            dby = by2-by
            for i in range(ifr):
                for var in var_dic:
                    real_flood[var].append(     dataset[index][1][var_dic[var]+4*i, bd:bd+nx, bd:bd+ny])
                    predicted_flood[var].append(dataset[index][1][var_dic[var]+4*i, bd:bd+nx, bd:bd+ny])
            
            for i in tqdm(range(st, et-(ifr-1)*tr, tr)):
                dx2 = torch.tensor([dataset[i*8 + augs][2]]).to(device, non_blocking=True)
                
                output = model(sx = sx, dx1 = dx1, dx2 = dx2, bc = bc, bc2 = bc2, norm_param = norm_param)
                
                if use_correction is False:
                    output_results = output["results"]
                else:
                    
                    if scene == 2:
                        steps = [1, 1, 0]
                        
                        qs = [[int(bc2[0][0]), int(bc2[0][1])]]
                        qs.append([0, 62])
                        qs.append([0, 63])
                        qs.append([0, 64])
                        r'''
                        qs = []
                        for qx in range(nx):
                            for qy in range(ny):
                                if predicted_flood["WD"][-1][qx, qy] > 0:
                                    qs.append([qx, qy])
                        '''      
                    else:
                        steps = [1, 1, 1]
                        qs = [[int(bc2[0][0]), int(bc2[0][1])]]
                        
                    output_results = find_connected_block(qs, output["results"], sx[0, 0, bd:bd+nx, bd:bd+ny], 
                                                          predicted_flood["WD"][-1], q = bc[0][2] + bc[0][3], dx2 = dx2, steps = steps, print_details = False)
                
                
                
                gg = output_results[0, :, :, :].cpu().numpy()
                for j in range(3):
                    flood_map[j][tot] = gg[j].reshape(nx*ny)
                tot += 1
                flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = eval_batch(output_results, np.expand_dims(dataset[i*8 + augs][2], 0))
                rmse_wd = math.sqrt(div(flSSE, flpix))
                rmse_q  = math.sqrt(div(torch.sum(torch.pow(output_results[0, 1:3, :, :] - dx2[0, 1:3, :, :], 2)), flpix*2))
                mae_wd  = div(torch.sum(abs(output_results[0, 0, :, :] - dx2[0, 0, :, :])), flpix)
                mae_q   = div(torch.sum(abs(output_results[0, 1:3, :, :] - dx2[0, 1:3, :, :])), flpix*2)
                        
                results.append({
                    "timestep": i+ifr*tr,
                    "RMSE-WD": float(rmse_wd), 
                    "RMSE-Q":  float(rmse_q), 
                    "MAE-WD":  float(mae_wd), 
                    "MAE-Q":   float(mae_q), 
                    "ratio of flooding area": div(l_H_005+l_FP_005, l_H_005+l_M_005), 
                    "AvgCSI": (div(l_H_005, (l_H_005+l_M_005+l_FP_005)) + div(l_H_030, (l_H_030+l_M_030+l_FP_030)))/2, 
                    "CSI_0.05": [div(l_H_005, (l_H_005+l_M_005+l_FP_005)), int(l_H_005), int(l_M_005), int(l_FP_005)], 
                    "CSI_0.30": [div(l_H_030, (l_H_030+l_M_030+l_FP_030)), int(l_H_030), int(l_M_030), int(l_FP_030)],
                    "Loss": float(output["loss"].detach().cpu().numpy()),
                    "ACC_0.05": div(l_ACC_005, l_TOTAL),
                    "ACC_0.30": div(l_ACC_030, l_TOTAL),
                    })
                print(" step:%d   RMSE-WD:%.4lf   AvgCSI:%.4lf    %d %d %d"%(i, results[-1]["RMSE-WD"], results[-1]["AvgCSI"], l_H_030, l_M_030, l_FP_030))
                print("           ", np.mean(real_flood["WD"][-1]), np.max(real_flood["WD"][-1]), np.min(real_flood["WD"][-1]))
                #print(" wd ", output_results[0, 0, :, 0])
                for j in range(ifr-1):
                    dx1[:, 4*j:4*(j+1), :, :] = dx1[:, 4*(j+1):4*(j+2), :, :]
                dx1[:, 4*(ifr-1):4*(ifr-1)+3, bd:bd+nx, bd:bd+ny] = output_results
                #if bd > 0:
                if scene == 0:
                    for k in range(bd):
                        dx1[0, 4*(ifr-1), bx-k*dbx, by-k*dby] = dx1[0, 4*(ifr-1), bx2, by2]
                elif scene == 2:
                    if bd == 1:
                        dx0 = torch.tensor([dataset[i*8 + augs][1]]).to(device, non_blocking=True)
                        dx1[0, 4*(ifr-1)+1, -1, bd+11:bd+14] = dx0[0, 4*(ifr-1)+1, -1, bd+11:bd+14]
                        dx1[0, 4*(ifr-1),   -1, bd+11:bd+14] = dx0[0, 4*(ifr-1),   -2, bd+11:bd+14]
                dx1[:, 4*(ifr-1)+3, :, :] = sx[:, 0, :, :] + dx1[:, 4*(ifr-1), :, :]
                
                losses = []
                loss_function = torch.nn.MSELoss()
                for j in range(output["results"].shape[1]):
                    losses.append(loss_function(dx2[:, j, :, :], output["results"][:, j, :, :]))

                for var in var_dic:
                    real_flood[var].append(     dx2[0, var_dic[var]].clone().cpu().numpy())
                    predicted_flood[var].append(dx1[0, var_dic[var]+4*(ifr-1), bd:bd+nx, bd:bd+ny].clone().cpu().numpy())
                results[-1]["sum_wd"] = [float(np.sum(real_flood["WD"][-1])), float(np.sum(predicted_flood["WD"][-1]))]
                
        
        print("-----overall scores-----")
        rmse_wd = 0
        rmse_q  = 0
        mae_wd  = 0
        mae_q   = 0
        csi = [[0, 0, 0, 0],
               [0, 0, 0, 0]]
        for rr in results:
            rmse_wd += (rr["RMSE-WD"]**2)
            rmse_q  += (rr["RMSE-Q"]**2)
            mae_wd  += rr["MAE-WD"]
            mae_q   += rr["MAE-Q"]
            for tt in range(1, 4):
                csi[0][tt] += rr["CSI_0.05"][tt]
                csi[1][tt] += rr["CSI_0.30"][tt]

        rmse_wd = math.sqrt(rmse_wd/len(results))
        rmse_q  = math.sqrt(rmse_q/len(results))
        mae_wd  /= len(results)
        mae_q   /= len(results)
        for cc in range(2):
            csi[cc][0] = csi[cc][1]/sum(csi[cc][1:])
        print("           RMSE-WD:%.4lf   AvgCSI:%.4lf-%.4lf-%.4lf    %d %d %d"%(rmse_wd, (csi[0][0]+csi[1][0])/2, csi[0][0], csi[1][0], csi[1][1], csi[1][2], csi[1][3]))
        results.append({
            "timestep": -1,
            "RMSE-WD": rmse_wd, 
            "RMSE-Q":  rmse_q, 
            "MAE-WD":  mae_wd, 
            "MAE-Q":   mae_q, 
            "AvgCSI": (csi[0][0]+csi[1][0])/2, 
            "CSI_0.05": csi[0], 
            "CSI_0.30": csi[1],
            })
        r''''''
        
        with open("%s/%s/results.json"%(results_path, ss), "w", encoding = "utf-8") as fout:
            json.dump(results, fout, indent = 4, ensure_ascii = False)
        for j in range(3):
            np.savetxt("%s/%s/results_%d.txt"%(results_path, ss, j), flood_map[j])
        
        if make_video:
            make_mp4(DEM = DEM, BC = BC, real_flood = real_flood, predicted_flood = [predicted_flood], model_list = [str(model_class)], var_list = ["WD", "QX", "QY"], st = st, tr = tr, save_path = "%s/%s/results.mp4"%(results_path, ss))

def work(args, results_list, excel_path = "results.xlsx", overwrite = False):
    import time
    excel = openpyxl.Workbook()
    sheet = excel.create_sheet(index=0, title="temp")
    
    sheet["A1"].value = "models"
    sheet["B1"].value = "RMSE-WD-overall"
    sheet["C1"].value = "RMSE-Q-overall"
    sheet["D1"].value = "MAE-WD-overall"
    sheet["E1"].value = "MAE-Q-overall"
    sheet["F1"].value = "AvgCSI-overall"
    sheet["G1"].value = "CSI0.05-overall"
    sheet["H1"].value = "CSI0.30-overall"
    sheet["I1"].value = "RMSE-WD-last frame"
    sheet["J1"].value = "RMSE-Q-last frame"
    sheet["K1"].value = "MAE-WD-last frame"
    sheet["L1"].value = "MAE-Q-last frame"
    sheet["M1"].value = "AvgCSI-last frame"
    sheet["N1"].value = "CSI0.05-last frame"
    sheet["O1"].value = "CSI0.30-last frame"
    sheet["P1"].value = "SumWD-last frame"
    sheet["Q1"].value = "Percentage of improved samples-h"
    sheet["R1"].value = "Percentage of improved samples-q"
    sheet["S1"].value = "Percentage of improved samples-csi"
    sheet["T1"].value = "Percentage of improved samples-all"
    
    test_samples = [
        [],
        [],
        [],
        [],
        []
    ]
    
    ss = 96
    #ss = 60
    
    for i in range(37, 41):
        for j in range(8):
            test_samples[0].append({"temporal_resolution": 2, "input_frames": 1, "start_timestep": 0, "end_timestep": ss, "seed": i, "augs": j})
            test_samples[1].append({"temporal_resolution": 2, "input_frames": 2, "start_timestep": 0, "end_timestep": ss, "seed": i, "augs": j})
    
    for i in range(500, 520):
        test_samples[2].append({"temporal_resolution": 2, "input_frames": 2, "start_timestep": 0, "end_timestep": 96, "seed": i, "augs": 0})
    
    test_samples[3].append({"temporal_resolution": 2, "input_frames": 1, "start_timestep": 0, "end_timestep": 24, "seed": 0, "augs": 0})
    test_samples[4].append({"temporal_resolution": 2, "input_frames": 2, "start_timestep": 0, "end_timestep": 24, "seed": 0, "augs": 0})
    
    tot = 2
    for dd in results_list:
        # model  ifr bd sc norm correct
        save_path, ifr, bd, sc, norm, corrected = dd
        model_test_path = "../models/%s"%(save_path)
        print("-----model: %s       ifr:%d   bd:%d   sc:%d   norm:%d   corrected:%d"%(model_test_path,  ifr, bd, sc, norm, corrected))
        
        pad_mode = 0
        grid_size = [64, 64]
        if sc == 2:
            pad_mode = 1
            grid_size = [97, 70]
        
        if "unet" in save_path:
            if "d3" in save_path:
                depth = 3
            else:
                depth = 4
            model_class = BaseCNN
            model = BaseCNN(ndepth = depth, bd = bd, ifr = ifr,
                            pad_mode = pad_mode,
                            grid_size = grid_size)    
            
        model.load_state_dict(torch.load(model_test_path))   
        
        if corrected is True or corrected > 1:
            save_path += "_corrected"
            if corrected > 1:
                save_path += "-%d"%(corrected)
        sheet["A%d"%(tot)].value = save_path
        
        args.input_frames = ifr
        args.boundary = bd
        if sc == 1:
            args.norm_param = "../data_cnn/norm_od.txt"
        elif sc == 2:
            args.norm_param = "../data_tous/norm_cnn.txt"
        elif norm == 0:
            args.norm_param = "../data_cnn/norm_cnn.txt"
        else:
            args.norm_param = "../data_cnn/norm_cnn2.txt"
            
        if sc == 1:
            testset = test_samples[2]
        elif sc == 2:
            if ifr == 1:
                testset = test_samples[3]
            else:
                testset = test_samples[4]
        elif ifr == 1:
            testset = test_samples[0]
        else:
            testset = test_samples[1]
        
        results_path = model_test_path.replace("/models/", "").replace("/", "--").replace("..", "../results_cnn/multistep_prediction_results/").replace(".bin", "")
        
        st = time.time()
        eval_multistep_prediction(args, model = model, model_test_path = model_test_path, model_class = model_class, test_samples = testset, results_path = results_path, device = torch.device("cuda"), use_correction = corrected, overwrite = overwrite)
        print(" cost time:%.4lf"%(time.time()-st))
        
        mm = np.zeros(21)
        if corrected is True or corrected > 1:
            results_path_og = results_path
            results_path += "_corrected"
            if corrected > 1:
                results_path += "-%d"%(corrected)
        
        improved_samples = [0, 0, 0, 0]
        
        for sample in testset:
            tr = sample["temporal_resolution"]
            st = sample["start_timestep"]
            et = sample["end_timestep"]
            ifr= sample["input_frames"]
            seed = sample["seed"]
            augs = sample["augs"]
            ss = "sd-%d tr-%d ifr-%d st-%d et-%d"%(seed, tr, ifr, st, et)
            if augs > 0:
                ss += " augs-%d"%(augs)
            
            with open("%s/%s/results.json"%(results_path, ss), "r", encoding = "utf-8") as fout:
                data = json.load(fout)
            if corrected is True or corrected > 1:
                with open("%s/%s/results.json"%(results_path_og, ss), "r", encoding = "utf-8") as fout:
                    data_og = json.load(fout)
            
                flag = 1
                temp = []
                for mv in ["RMSE-WD", "RMSE-Q", "MAE-WD", "MAE-Q", "CSI_0.05", "CSI_0.30"]:
                    v1 = data_og[-1][mv]
                    v2 = data[-1][mv]
                    if "CSI" in mv:
                        v1 = -v1[0]
                        v2 = -v2[0]
                    if v1 <= v2:
                        flag = 0
                    temp.append(v1 > v2)
                if temp[0] == True and temp[2] == True:
                    improved_samples[0] += 1
                if temp[1] == True and temp[3] == True:
                    improved_samples[1] += 1
                if temp[4] == True and temp[5] == True:
                    improved_samples[2] += 1
                if flag == 1:
                    improved_samples[-1] += 1
                
            mm[0] += data[-1]["RMSE-WD"]**2
            mm[1] += data[-1]["RMSE-Q"]**2
            mm[2] += data[-1]["MAE-WD"]
            mm[3] += data[-1]["MAE-Q"]
            mm[10] += data[-2]["RMSE-WD"]**2
            mm[11] += data[-2]["RMSE-Q"]**2
            mm[12] += data[-2]["MAE-WD"]
            mm[13] += data[-2]["MAE-Q"]
            for j in range(3):
                mm[4+j] += data[-1]["CSI_0.05"][1+j]
                mm[7+j] += data[-1]["CSI_0.30"][1+j]
                mm[14+j] += data[-2]["CSI_0.05"][1+j]
                mm[17+j]+= data[-2]["CSI_0.30"][1+j]
            mm[20] += abs(data[-2]["sum_wd"][1] - data[-2]["sum_wd"][0]) / data[-2]["sum_wd"][0]
        rmse00 = math.sqrt(mm[0]/len(testset))
        rmse01 = math.sqrt(mm[1]/len(testset))
        mae00  = mm[2]/len(testset)
        mae01  = mm[3]/len(testset)
        rmse10 = math.sqrt(mm[10]/len(testset))
        rmse11 = math.sqrt(mm[11]/len(testset))
        mae10  = mm[12]/len(testset)
        mae11  = mm[13]/len(testset)
        
        csi50 = mm[4]/sum(mm[4:7])
        csi30 = mm[7]/sum(mm[7:10])
        csi51 = mm[14]/sum(mm[14:17])
        csi31 = mm[17]/sum(mm[17:20])
        swd   = mm[20]/len(testset)
        vv = [rmse00, rmse01, mae00, mae01, (csi50+csi30)/2, csi50, csi30, rmse10, rmse11, mae10, mae11, (csi51+csi31)/2, csi51, csi31, swd]
        for j in range(15):
            sheet["%s%d"%(chr(ord("B")+j), tot)].value = vv[j]
        if corrected is True or corrected > 1:   
            for j in range(15, 19):
                sheet["%s%d"%(chr(ord("B")+j), tot)].value = improved_samples[j-15] / len(testset)
        
        tot += 1 
        excel.save(excel_path)

if __name__=="__main__":
    
    args = parser.parse_args()
    args.eval_batch_size = 1#4
    
    args.temporal_resolution = 2
    args.simulation_steps = 120
    args.train_data_seed = range(1, 37)
    args.valid_data_seed = range(33, 37)
    #args.valid_data_seed = range(37, 41)
    
    
    args.src_loss   = True
    
    model_class = BaseCNN

    # setting for synthetic dataset
    args.boundary = 1
    args.input_frames = 1
    args.scene  = 0
    args.norm_param = "../data_cnn/norm_cnn.txt"
    model_test_path = "../models/unet-d4-htp.bin"
    
    r'''
    # setting for Tous dam break dataset
    args.boundary = 1
    args.input_frames = 1
    args.simulation_steps = 80
    args.scene  = 2
    args.augs   = False
    args.norm_param = "../data_tous/norm_cnn.txt"
    '''
    
    #eval_valid(args = args, model_test_path = model_test_path, model_class = model_class, results_path = model_test_path.replace("/models/", "").replace("..", "../results_cnn/valid_results/").replace(".bin", ".json"), device = torch.device("cuda"), overwrite = True)
    
    
    r'''
    # prediction for a specific sample with a trained model
    test_samples = []
    test_samples.append({"temporal_resolution": 2, "input_frames": 2, "start_timestep": 0, "end_timestep": 96, "seed": 39, "augs": 0})
    eval_multistep_prediction(args, model_test_path = model_test_path, model_class = model_class, test_samples = test_samples, results_path = model_test_path.replace("/models/", "").replace("..", "../results_cnn/multistep_prediction_results/").replace(".bin", ""), device = torch.device("cuda"), use_correction = False, overwrite = True, make_video = False)
    '''
    
    
    # generate prediction results
    # format: [model, input_frame, boundary, scene, normalization, correction]
    # sfp = single frame prediction    htp = historical trend prediction
    results_list = [
        #Table 1
        ["unet-d3-sfp.bin", 1, 1, 0, 0, False],
        ["unet-d4-sfp.bin", 1, 1, 0, 0, False],
        ["unet-d3-htp.bin", 2, 0, 0, 0, False], 
        ["unet-d4-htp.bin", 2, 0, 0, 0, False], 
        
        ["unet-d3-sfp.bin", 1, 1, 0, 0, True],
        ["unet-d4-sfp.bin", 1, 1, 0, 0, True],
        ["unet-d3-htp.bin", 2, 0, 0, 0, True], 
        ["unet-d4-htp.bin", 2, 0, 0, 0, True], 
        
        #Table 4 initial period
        ["unet-d4-tous-sfp.bin", 1, 1, 2, 0, False], 
        ["unet-d4-tous-sfp.bin", 1, 1, 2, 0, True], 
        ["unet-d4-tous-htp.bin", 2, 0, 2, 0, False],
        ["unet-d4-tous-htp.bin", 2, 0, 2, 0, True],
        ]
    work(args, results_list, excel_path = "results.xlsx", overwrite = False)
