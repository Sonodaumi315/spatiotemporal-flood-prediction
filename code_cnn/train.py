import argparse
import json
import os
from tqdm.auto import tqdm
import time
import numpy as np
import datetime
import math
import random
import torch
from torch.optim import AdamW
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

from model_unet import BaseCNN
from utils import get_num, reduce_sum
from dataset import FloodDatasetCNN, make_dataset
from eval import evaluate, eval_batch, cal

parser = argparse.ArgumentParser()
parser.add_argument('--dist', type=bool, default=False,
                help='distributed training')
parser.add_argument('--local_rank', type=int, default=-1,
                help='node rank for distributed training')

parser.add_argument('--model_load_path', type=str, default = None,
                help='model load path')
parser.add_argument('--model_save_path', type=str, default = None,
                help='model save path')

parser.add_argument('--accumulation', type=int, default=1,
                help='gradient accumulation steps')
parser.add_argument('--max_lr', type=float, default=1e-3,#1e-5,
                help='max learning rate')
parser.add_argument('--min_lr', type=float, default=5e-5,#5e-7,
                help='min learning rate')
parser.add_argument('--max_norm', type=float, default=1.0,
                help=" clipping gradient norm")
parser.add_argument('--epochs', type=int, default=100,
                help='training epochs')
parser.add_argument('--batch_size', type=int, default=4,
                help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=4,
                help='eval batch size')
parser.add_argument('--valid_interval', type=int, default=1,
                help='the interval between 2 validations')
parser.add_argument('--valid_per_epoch', type=int, default=5,
                    help='valid per epoch (when valid_interval=1)')
parser.add_argument('--save_interval', type=int, default=5,
                help='the interval between 2 saved models')

parser.add_argument('--schedule_var1', type=int, default=0,
                help='schedule var1')
parser.add_argument('--schedule_var2', type=int, default=0,
                help='schedule var2')
parser.add_argument('--seed', type=int, default=3407,#233,
                help='random seed')
parser.add_argument('--wd', type=float, default=1e-2,
                help='weight decay')

parser.add_argument('--schedule', type=int, default=0,
                help='type of schedule  0:cosine   1:linear   2:cycle')
parser.add_argument('--lr_step_interval', type=int, default=0,
                help='0: per epoch   1: per batch')
parser.add_argument('--augs', type=bool, default=False,
                help='data augmentation')
parser.add_argument('--scene', type=int, default=0,
                help='0:synthetic  1:bentivogilio  2:tous dam')
parser.add_argument('--rollout', type=int, default=1,
                help='rollout')

parser.add_argument('--temporal_resolution', type=int, default=2,
                help='temporal resolution')
parser.add_argument('--simulation_steps', type=int, default=120,
                help='simulation steps')
parser.add_argument('--input_frames', type=int, default=2,
                help='input frames')
parser.add_argument('--boundary', type=int, default=1,
                help='boundary')
parser.add_argument('--sf', type=int, default=3,
                help='static_features')
parser.add_argument('--df', type=int, default=4,
                help='dynamic_features')

parser.add_argument('--base_filters', type=int, default=64,
                help='base_filters')
        
parser.add_argument('--train_data_seed', type=list, default=None,
                help='train data seed')
parser.add_argument('--valid_data_seed', type=list, default=None,
                help='valid data seed')
parser.add_argument('--norm_param', type=str, default=None,
                help='normalization parameters')

def main_worker(local_rank, nprocs, args, model_class):
    print(" start main work  local_rank:%d  nprocs:%d"%(local_rank, nprocs))
    
    if args.dist:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl')
    else:
        device = torch.device("cuda")
    cudnn.benchmark = True
    
    #get parameters 
    num_epochs      = args.epochs
    batch_size      = args.batch_size
    eval_batch_size = args.eval_batch_size
    accumulation    = args.accumulation
    valid_interval  = args.valid_interval
    save_interval   = args.save_interval
    
    model_save_path = args.model_save_path
    if model_save_path is None:
        model_save_path = "%s %s"%(str(datetime.date.today()),str(datetime.datetime.now().time())[:8].replace(":","-"))
    model_load_path = args.model_load_path
    
    train_losses = []
    valid_losses = []
    #create model
    simulation_steps    = args.simulation_steps
    input_frames        = args.input_frames
    temporal_resolution = args.temporal_resolution
    
    if args.scene == 2:
        pad_mode = 1
        grid_size = [97, 70]
    else:
        pad_mode = 0
        grid_size = [64, 64]
        
    model = model_class(base_filters = args.base_filters, bd = args.boundary, ifr = input_frames, 
                        pad_mode = pad_mode,
                        grid_size = grid_size)
        
    if args.model_load_path:
        model.load_state_dict(torch.load(model_load_path))
    
    if args.dist:
        print(" model convert_sync_batchnorm ")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model.to(device)
    
    if args.dist:
        args.batch_size      = args.batch_size // nprocs
        args.eval_batch_size = args.eval_batch_size // nprocs
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])#, find_unused_parameters=True)
    
    train_data_seed     = args.train_data_seed
    valid_data_seed     = args.valid_data_seed
    
    
    train_dataset = "../data_cnn/train_%d-%d"%(train_data_seed[0], train_data_seed[-1])
    valid_dataset = "../data_cnn/valid_%d-%d"%(valid_data_seed[0], valid_data_seed[-1])
    if args.scene is True:
        args.augs = False
    if args.augs is True:
        train_dataset += "_augs"
        valid_dataset += "_augs"
    
    if args.norm_param is None:
        norm_param = None
    else:
        norm_param = np.loadtxt(args.norm_param)
    
    
    if args.dist:
        train_dataset, train_steps, train_sampler = make_dataset(save_path = train_dataset, seed = train_data_seed, simulation_steps = simulation_steps, input_frames = input_frames, temporal_resolution = temporal_resolution, 
                                                                 shuffle = False, boundary = args.boundary, augs = args.augs, scene = args.scene, dist = args.dist, nprocs = nprocs, batch_size = batch_size) 
        valid_dataset, valid_steps, valid_sampler = make_dataset(save_path = valid_dataset, seed = valid_data_seed, simulation_steps = simulation_steps, input_frames = input_frames, temporal_resolution = temporal_resolution, 
                                                                 shuffle = False, boundary = args.boundary, augs = args.augs, scene = args.scene, dist = args.dist, nprocs = nprocs, batch_size = eval_batch_size)
    else:
        train_dataset, train_steps                = make_dataset(save_path = train_dataset, seed = train_data_seed, simulation_steps = simulation_steps, input_frames = input_frames, temporal_resolution = temporal_resolution, 
                                                                 shuffle = True,  boundary = args.boundary, augs = args.augs, scene = args.scene, batch_size = batch_size)
        valid_dataset, valid_steps                = make_dataset(save_path = valid_dataset, seed = valid_data_seed, simulation_steps = simulation_steps, input_frames = input_frames, temporal_resolution = temporal_resolution, 
                                                                 shuffle = False, boundary = args.boundary, augs = args.augs, scene = args.scene , batch_size = eval_batch_size)
        
    num_train_data  = len(train_dataset)
    num_valid_data  = len(valid_dataset)
    train_eval_num  = 1e9#max(1, 200 // batch_size)
    
    optimizer_class = AdamW
    if (args.dist == False) or (local_rank == 0):
        print("   use Adamw as optimizer")
    
    if args.schedule == 0:
        print("---------use CosineAnnealingLR")
        optimizer = optimizer_class(filter(lambda p:p.requires_grad, model.parameters()), lr=args.max_lr, weight_decay=args.wd)
        if args.lr_step_interval == 0:
            lr_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.schedule_var1,                eta_min = args.min_lr)
        else:
            lr_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.schedule_var1*num_train_data, eta_min = args.min_lr)
    elif args.schedule == 1: 
        print("---------use linear_schedule")
        optimizer = optimizer_class(filter(lambda p:p.requires_grad, model.parameters()), lr=1, weight_decay=args.wd)
        def linear_schedule(up_lr=5e-3, lo_lr=1e-5, schedule_var1 = 0, schedule_var2 = 0, num_epochs = 250):
            def schedule(epoch):
                if epoch < schedule_var1:
                    lr = up_lr
                elif epoch < schedule_var2:
                    lr = up_lr - (up_lr-lo_lr)*(epoch-schedule_var1)/(schedule_var2-schedule_var1)
                else:
                    lr = lo_lr
                return (lr)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule, last_epoch=-1)
        if args.lr_step_interval == 0:
            lr_scheduler = linear_schedule(up_lr = args.max_lr, lo_lr = args.min_lr, schedule_var1 = args.schedule_var1, 
                                           schedule_var2 = args.schedule_var2,                num_epochs = num_epochs)
        else:
            lr_scheduler = linear_schedule(up_lr = args.max_lr, lo_lr = args.min_lr, schedule_var1 = args.schedule_var1*num_train_data, 
                                           schedule_var2 = args.schedule_var2*num_train_data, num_epochs = num_epochs*num_train_data)
    elif args.schedule == 2:
        print("---------use cycle_schedule")
        optimizer = optimizer_class(filter(lambda p:p.requires_grad, model.parameters()), lr=1, weight_decay=args.wd)
        def step_decay_schedule(up_lr=5e-3, lo_lr=1e-4, decay_factor=0.9, stepsize=5, work_epochs = 500, basepoch=0):
            def schedule(epoch):
                epoch2=epoch+basepoch
                if epoch2<work_epochs:
                    cycle=np.floor(1+epoch2/(2*stepsize))
                    lrf=1-np.abs(epoch2/stepsize-2*cycle+1)
                    lr=(up_lr-lo_lr)*lrf*np.power(decay_factor,epoch2/stepsize)+lo_lr
                else:
                    lr=lo_lr
                return (lr)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule, last_epoch=-1)
        if args.lr_step_interval == 0:
            lr_scheduler = step_decay_schedule(up_lr = args.max_lr, lo_lr = args.min_lr, decay_factor=0.95, stepsize=10, 
                                               work_epochs=num_epochs)
        else:
            lr_scheduler = step_decay_schedule(up_lr = args.max_lr, lo_lr = args.min_lr, decay_factor=0.95, stepsize=10*num_train_data, 
                                               work_epochs=num_epochs*num_train_data)
    
    if args.augs is True:
        train_eval_num = train_eval_num*8
        
    if local_rank <= 0:
        print(" num_train_data: %d"%(num_train_data))
        print(" num_valid_data: %d"%(num_valid_data))
        print(" train_steps: %d"%(train_steps))
        print(" valid_steps: %d"%(valid_steps))
        
        print(" batch_size: %d"%(batch_size))
        print(" eval_batch_size: %d"%(eval_batch_size))
        print(" accumulation: %d"%(accumulation))
        print(" model_class: %s"%(str(model_class)))
        print(" augs: %d"%(args.augs))
        print(" max_lr: %.8lf"%(args.max_lr))
        print(" min_lr: %.8lf"%(args.min_lr))
        print(" weight decay: ", args.wd)
        print(" schedule: ", args.schedule)
        print(" lr_step_interval: %d"%(args.lr_step_interval))
        print(" epochs: %d"%(num_epochs))
        print(" schedule_var1: %d"%(args.schedule_var1))
        print(" schedule_var2: %d"%(args.schedule_var2))
        print(" normalization parameters::%s\n "%(args.norm_param), norm_param)
        print(" boundary: ", args.boundary)
        print(" scene: ", args.scene)
        
        print(" model save path: %s"%(model_save_path))
        print(" model load path: %s"%(model_load_path))
        
        print(" train_eval_num: %d"%(train_eval_num))
        
        if not os.path.exists("../models/%s"%(model_save_path)):
            os.makedirs("../models/%s"%(model_save_path))
            
    valid_batch = [num_train_data]
    if args.valid_interval == 1:
        a = num_train_data // args.valid_per_epoch
        for i in range(1, args.valid_per_epoch):
            valid_batch.append(int(i*a))
    print(" valid_batch ", valid_batch)
    
    best_model = 1e9
    
    for epoch in tqdm(range(1,num_epochs+1)):
        if args.dist:
            train_sampler.set_epoch(epoch+1)
        
        model.train()
        cnt_valid = 0
        
        l_ACC_005 = 0; l_ACC_030 = 0
        l_H_005   = 0; l_H_030   = 0
        l_M_005   = 0; l_M_030   = 0
        l_FP_005  = 0; l_FP_030  = 0
        l_TOTAL   = 0
        flSSE     = 0
        flpix     = 0
        tot_loss_train = 0
        tot_train      = 0
        
        for batch in tqdm(train_dataset):
            sx  = batch[0].to(device, non_blocking=True)
            dx1 = batch[1].to(device, non_blocking=True)
            dx2 = batch[2].to(device, non_blocking=True)
            bc  = batch[3].to(device, non_blocking=True)
            bc2 = batch[4].to(device, non_blocking=True)

            output = model(sx = sx, dx1 = dx1, dx2 = dx2, bc = bc, bc2 = bc2, norm_param = norm_param)
            loss = output["loss"] / float(accumulation)
            
            tot_train += 1
            tot_loss_train += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            
            if tot_train <= train_eval_num:
                flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = eval_batch(output["results"], batch[2],
                flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL)

            if tot_train % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            if (epoch%valid_interval == 0) and (tot_train in valid_batch):
                cnt_valid += 1
                valid_loss = evaluate(prefix = "valid", model = model, norm_param = norm_param, val_generator = valid_dataset, local_rank = local_rank, return_loss = True)
                
                if valid_loss < best_model:
                    best_model = valid_loss
                    if local_rank <= 0:
                        if args.dist == False:
                            save_model = model.state_dict()
                        else:
                            save_model = model.module.state_dict()
                        torch.save(save_model, "../models/%s/best_model.bin"%(model_save_path))
                        
                if local_rank <= 0:
                    a = get_num(epoch)
                    if args.dist == False:
                        save_model = model.state_dict()
                    else:
                        save_model = model.module.state_dict()
                    torch.save(save_model, "../models/%s/ep-%s-%d.bin"%(model_save_path,a,cnt_valid))    

                    print("---epoch %d/%d  valid_loss:%.5lf  best_model_loss:%.5lf"%(epoch, num_epochs, valid_loss, best_model))
                    valid_losses.append(valid_loss)
                    with open("../models/%s/valid_losses.json"%(model_save_path),"w",encoding='utf-8') as fout:
                        json.dump(valid_losses,fout,indent=4,ensure_ascii=False)  
                        
                model.train()
                
            if args.lr_step_interval == 1: 
                lr_scheduler.step()

        if args.lr_step_interval == 0: 
            lr_scheduler.step()
        
        r''''''
        if args.dist:
            l_ACC_005 = torch.tensor(l_ACC_005, dtype=float).to(device); l_ACC_030 = torch.tensor(l_ACC_030, dtype=float).to(device)
            l_H_005   = torch.tensor(l_H_005, dtype=float).to(device);   l_H_030   = torch.tensor(l_H_030, dtype=float).to(device)
            l_M_005   = torch.tensor(l_M_005, dtype=float).to(device);   l_M_030   = torch.tensor(l_M_030, dtype=float).to(device)
            l_FP_005  = torch.tensor(l_FP_005, dtype=float).to(device);  l_FP_030  = torch.tensor(l_FP_030, dtype=float).to(device)
            l_TOTAL   = torch.tensor(l_TOTAL, dtype=float).to(device)
            flSSE     = torch.tensor(flSSE, dtype=float).to(device)
            flpix     = torch.tensor(flpix, dtype=float).to(device)
            tot_loss_train = torch.tensor(tot_loss_train, dtype=float).to(device)
            tot_train      = torch.tensor(tot_train, dtype=float).to(device)
            l_ACC_005 = reduce_sum(l_ACC_005); l_ACC_030 = reduce_sum(l_ACC_030)
            l_H_005   = reduce_sum(l_H_005);   l_H_030   = reduce_sum(l_H_030)
            l_M_005   = reduce_sum(l_M_005);   l_M_030   = reduce_sum(l_M_030)
            l_FP_005  = reduce_sum(l_FP_005);  l_FP_030  = reduce_sum(l_FP_030)
            l_TOTAL   = reduce_sum(l_TOTAL)
            flSSE     = reduce_sum(flSSE)
            flpix     = reduce_sum(flpix)
            tot_loss_train = reduce_sum(tot_loss_train)
            tot_train      = reduce_sum(tot_train)
            dist.barrier()
        
        tot_loss_train /= tot_train
        
        if local_rank <= 0:
            train_losses.append(float(tot_loss_train.detach().cpu().numpy()))
            cal("train", tot_loss_train, flSSE, flpix, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, False)
            print("###epoch %d/%d  train_loss:%.5lf/%.5lf  lr=%.8lf"%(epoch, num_epochs, tot_loss_train, math.sqrt(tot_loss_train), optimizer.param_groups[0]["lr"]))
            
            if epoch%save_interval==0:
                with open("../models/%s/train_losses.json"%(model_save_path),"w",encoding='utf-8') as fout:
                    json.dump(train_losses, fout, indent=4, ensure_ascii=False)
        
def setup_seed(seed):
    print(" seed:%d"%(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.deterministic = False

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

if __name__=="__main__":
    
    print(" start main ")
    
    args = parser.parse_args()
    args.seed = 666
    
    args.max_lr = 1e-3
    args.min_lr = 1e-5
    args.epochs = 300
    args.schedule_var1 = 10
    args.schedule_var2 = 101
    args.wd = 1e-2
    args.schedule = 0
    args.augs   = True
    args.lr_step_interval = 0
    args.valid_per_epoch  = 1
    args.valid_interval   = 1
    args.save_interval    = 1
    
    args.temporal_resolution = 2
    args.simulation_steps = 120
    
    args.accumulation = 1
    model_class = BaseCNN
    
    
    # for historical trend prediction 
    args.input_frames = 2
    args.boundary = 0
    
    # for single frame prediction 
    args.input_frames = 1
    args.boundary = 1
    
    
    # for synthetic dataset
    args.norm_param = "../data_cnn/norm_cnn.txt"
    args.train_data_seed = range(1, 33)
    args.valid_data_seed = range(33, 37)
    args.batch_size   = int(128)
    args.eval_batch_size = int(32)
    args.scene = 0
    
    
    # for tous dam break dataset
    r'''
    args.epochs = 500
    args.norm_param = "../data_tous/norm_cnn.txt"
    args.train_data_seed = range(1, 9)
    args.valid_data_seed = [9]
    args.batch_size   = int(8)
    args.eval_batch_size = int(1)
    args.simulation_steps = 80
    args.scene  = 2
    args.augs   = False
    '''
    
    use_gpu     = True
    
    args.dist = (args.local_rank != -1)
    args.nprocs = torch.cuda.device_count()
    if args.seed is not None:
        setup_seed(args.seed)
    main_worker(args.local_rank, args.nprocs, args, model_class)   