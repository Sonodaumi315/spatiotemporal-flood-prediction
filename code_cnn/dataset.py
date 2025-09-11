import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

class FloodDatasetCNN(Dataset):
    
    def __init__(self, seed, simulation_steps, input_frames = 2, temporal_resolution = 2, static_features = 3, dynamic_features = 4, augs = 0, bd = 0, scene = 0):
        
        self.event = []
        self.augs  = augs
        self.seed  = seed
        self.simulation_steps = simulation_steps
        self.input_frames = input_frames
        self.temporal_resolution = temporal_resolution
        self.static_features  = static_features       #dem slpx slpy
        self.dynamic_features = dynamic_features      #wd qx qy e
        self.bd = bd
        self.sc = scene
        
        ss  = self.simulation_steps
        ifr = self.input_frames
        tr  = self.temporal_resolution
        
        
        self.len = len(seed)*(ss-tr*ifr+1)

        if self.augs is True:
            self.len = self.len*8
        
        self.var_list = ["BC", "dem", "slpx", "slpy", "wd", "qx", "qy"]
            
        print("----make dataset----")
        
        if scene == 0: # synthetic dataset
            for sd in seed:
                print(" seed ", sd)
                
                temp = {}
                for dd in self.var_list:
                    try:
                        temp[dd] = np.loadtxt("../data_cnn/%s/%d.txt"%(dd, sd))
                    except:
                        temp[dd] = np.zeros((ss+1, 68, 68))
                        for i in range(ss+1):
                            temp[dd][i] = np.loadtxt("../data_cnn/%s/%d_%d.txt"%(dd, sd, i))
                    
                temp["e"] = np.zeros((ss+1, 68, 68))
                dem = temp["dem"]        
                for i in range(ss+1):
                    wd = temp["wd"][i]
                    temp["e"][i] = dem + wd
                
                self.event.append(temp)
        elif scene == 1: # dataset created by Bentivoglio et al. (2023), could not be used in this study
            self.simulation_steps = 96
            ss  = 96
            for sd in seed:
                print(" seed ", sd)
                temp = {}
                temp["dem"] = np.zeros((68, 68))
                gg = np.loadtxt("../data/Bentivoglio2023/raw_datasets/DEM/DEM_%d.txt"%(sd))[:, 2]
                temp["dem"][2:66, 2:66] = gg.reshape(64, 64)
                temp["dem"] -= np.min(temp["dem"])

                slpx, slpy = torch.gradient(torch.tensor(temp["dem"][2:66, 2:66], dtype=torch.float32))
                temp["slpx"] = np.zeros((68, 68))
                temp["slpx"][2:66, 2:66] = slpx.numpy()
                temp["slpy"] = np.zeros((68, 68))
                temp["slpy"][2:66, 2:66] = slpy.numpy()
                
                temp["qx"] = np.zeros((ss+1, 68, 68))
                gg = np.loadtxt("../data/Bentivoglio2023/raw_datasets/VX/VX_%d.txt"%(sd))
                temp["qx"][:, 2:66, 2:66] = gg.reshape(ss+1, 64, 64)
                
                temp["qy"] = np.zeros((ss+1, 68, 68))
                gg = np.loadtxt("../data/Bentivoglio2023/raw_datasets/VY/VY_%d.txt"%(sd))
                temp["qy"][:, 2:66, 2:66] = gg.reshape(ss+1, 64, 64)
                
                temp["wd"] = np.zeros((ss+1, 68, 68))
                gg = np.loadtxt("../data/Bentivoglio2023/raw_datasets/WD/WD_%d.txt"%(sd))
                temp["wd"][:, 2:66, 2:66] = gg.reshape(ss+1, 64, 64)
                    
                temp["e"] = np.zeros((ss+1, 68, 68))
                dem = temp["dem"]        
                for i in range(ss+1):
                    wd = temp["wd"][i]
                    temp["e"][i] = dem + wd
                
                temp["BC"] = [0, 0, 0, 0.5]
                
                self.event.append(temp)
        elif scene == 2: # Tous dam break dataset
            for sd in seed:
                print(" seed ", sd)
                
                temp = {}
                for dd in self.var_list:
                    try:
                        temp[dd] = np.loadtxt("../data_tous/%s/%d.txt"%(dd, sd))
                    except:
                        temp[dd] = np.zeros((ss+1, 101, 74), dtype = np.float32)
                        for i in range(ss+1):
                            temp[dd][i] = np.loadtxt("../data_tous/%s/%d_%d.txt"%(dd, sd, i))
                        
                temp["dem"] = np.flip(temp["dem"], 0)
                temp["e"] = np.zeros((ss+1, 101, 74))
                dem = temp["dem"]        
                for i in range(ss+1):
                    wd = temp["wd"][i]
                    temp["e"][i] = dem + wd
                
                self.event.append(temp)
        
        self.get_info()
            
    def get_info(self):
        print(" ----make FloodDatasetCNN----")
        print("  seed: ", self.seed)
        print("  simulation steps: ", self.simulation_steps)
        print("  temporal resolution: ", self.temporal_resolution)
        print("  scene", self.sc)
        print("  input frames: ", self.input_frames)
        print("  static_features: ", self.static_features)
        print("  dynamic_features: ", self.dynamic_features)
        print("  bd: ", self.bd)
        print("  augs: ", self.augs)
    
    def __len__(self):
        return self.len

    def rotate(self, bc):
        ct = (64-1) / 2.0
        x0 = bc[0] - ct
        y0 = bc[1] - ct
        qx0 = bc[2]
        qy0 = bc[3]
        
        bc[0] = -y0 + ct
        bc[1] = x0 + ct
        bc[2] = -qy0
        bc[3] = qx0
        return bc
    
    def __getitem__(self, index):
        if self.augs == True:
            rot   = (index % 8) %  4
            flp   = (index % 8) // 4
            index = index // 8

        ss  = self.simulation_steps
        ifr = self.input_frames
        tr  = self.temporal_resolution
        bd  = self.bd
        sx_index = index // (ss-tr*ifr+1)
        dx_index = index %  (ss-tr*ifr+1)
        
        if self.sc == 2:
            nx, ny = 97, 70
        else:
            nx, ny = 64, 64

        sx  = np.zeros((self.static_features,      nx+2*bd, ny+2*bd), dtype = np.float32) #static features  (z, slpx, slpy)
        dx1 = np.zeros((ifr*self.dynamic_features, nx+2*bd, ny+2*bd), dtype = np.float32) #dynamic features (h, qx, qy, e) inputs
        dx2 = np.zeros((self.dynamic_features,     nx, ny), dtype = np.float32)           #dynamic features (h, qx, qy, e) outputs
        bc  = np.arange(4, dtype = np.float32)
        bc2 = np.arange(4, dtype = np.float32)
        
        sx_vars = ["dem", "slpx", "slpy"]
        dx_vars = ["wd", "qx", "qy", "e"]
        
        for i in range(len(sx_vars)):
            d = sx_vars[i]
            sx[i] = self.event[sx_index][d][2-bd:2+nx+bd, 2-bd:2+ny+bd]
        # boundary condition setting
        if self.sc == 2:
            bc[2] = (self.event[sx_index]["BC"][dx_index] + self.event[sx_index]["BC"][dx_index + tr*ifr])/2/ifr/tr
            tempbc = (self.event[sx_index]["qx"][dx_index][2+nx][13] + self.event[sx_index]["qx"][dx_index + tr*ifr][2+nx][13])/2/ifr/tr
            for j in range(1, ifr*tr):
                bc[2] += self.event[sx_index]["BC"][dx_index + j]/ifr/tr
                tempbc += self.event[sx_index]["qx"][dx_index + j][2+nx][13]/ifr/tr
            bc[2] = bc[2]/200
            bc2[0] = nx-1
            bc2[1] = 12
            bc2[2] = bc[2]
        else:
            for i in range(4):
                bc[i] = self.event[sx_index]["BC"][i]
                bc2[i]= self.event[sx_index]["BC"][i]
            
        for i in range(len(dx_vars)):
            d = dx_vars[i]
            for j in range(ifr):
                dx1[i+j*len(dx_vars)] = self.event[sx_index][d][dx_index + tr*j][2-bd:2+nx+bd, 2-bd:2+ny+bd]
            dx2[i] = self.event[sx_index][d][dx_index + tr*ifr][2:2+nx, 2:2+ny]
        
        bx, by =int(bc[0]+self.bd), int(bc[1]+self.bd)
        
        if self.sc == 2:
            for k in range(bd):
                for j in range(bd+11, bd+14):
                    dx1[1][bd+nx+k][j] = bc[2]
                for j in range(bd+62, bd+65):
                    dx1[0][k][j] = 5
                    dx1[3][k][j] = 45
        else:
            if dx_index == 0:
                for k in range(bd):
                    dx1[2][bx][by-k] = bc[3]
            bc2[1] += 1
            
        if self.augs == True: # data augmentation
            # Rotate counterclockwise by 90 degrees
            for i in range(rot):
                sx  = np.rot90(sx,  1, [1, 2])
                dx1 = np.rot90(dx1, 1, [1, 2])
                dx2 = np.rot90(dx2, 1, [1, 2])
                bc = self.rotate(bc)
                bc2 = self.rotate(bc2)
                
                a = 1
                b = 2
                sx[[a, b], :, :]  = sx[[b, a], :, :]
                for j in range(ifr):
                    dx1[[a+j*len(dx_vars), b+j*len(dx_vars)], :, :] = dx1[[b+j*len(dx_vars), a+j*len(dx_vars)], :, :]
                dx2[[a, b], :, :] = dx2[[b, a], :, :]
                sx[a, :, :]  *= -1
                for j in range(ifr):
                    dx1[a+j*len(dx_vars), :, :] *= -1
                dx2[a, :, :] *= -1
            # flipping
            if flp == 1:
                sx  = np.flip(sx,  1)
                dx1 = np.flip(dx1, 1)
                dx2 = np.flip(dx2, 1)
                
                sx[1, :, :]  *= -1
                for j in range(ifr):
                    dx1[1+j*len(dx_vars), :, :] *= -1
                dx2[1, :, :] *= -1
                
                bc[0] = 64 - 1 - bc[0]
                bc2[0]= 64 - 1 - bc2[0]
                #bc[2] = -bc[2]
            
            sx  = np.ascontiguousarray(sx)
            dx1 = np.ascontiguousarray(dx1)
            dx2 = np.ascontiguousarray(dx2)
            
        return (sx, dx1, dx2, bc, bc2)
        
def make_dataset(save_path, seed, batch_size = 16, simulation_steps = 120, input_frames = 2, temporal_resolution = 2, boundary = 1,
                 shuffle = False, save_cache = True, augs = True, scene = 0, dist = False, nprocs = 1):


    if scene == 1:
        save_path += "-od"
        boundary = 0
        simulation_steps = 96
    elif scene == 2:
        save_path += "-tous"
        simulation_steps = 80
    save_path += "-%d"%(1+simulation_steps-input_frames*temporal_resolution)
    save_path += "-bd%d"%(boundary)
    
    if os.path.isfile(save_path):
        print(" load dataset_cache:%s"%(save_path))
        dataset = torch.load(save_path)
    else:
        print(" dataset_cache:%s not exists"%(save_path))
        dataset = FloodDatasetCNN(seed = seed, simulation_steps = simulation_steps, input_frames = input_frames, temporal_resolution = temporal_resolution, bd = boundary, augs = augs, scene = scene)
        if save_cache:
            torch.save(dataset, save_path, pickle_protocol = 4)
    
    steps = len(dataset)
    print(" shuffle ", shuffle)
    if dist == True:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, shuffle = shuffle, batch_size = batch_size, drop_last = True, num_workers = 4, pin_memory = True, sampler = sampler)
        steps = steps // (batch_size*nprocs)
    else:
        dataloader = DataLoader(dataset, shuffle = shuffle, batch_size = batch_size, drop_last = True)
        steps = steps // batch_size
        
    print(" %s  len:%d=batch_size:%d x steps:%d"%(save_path, batch_size*steps, batch_size, steps))
    
    if dist == True:
        return dataloader, steps, sampler
    else:
        return dataloader, steps

if __name__ == "__main__":
    #make_dataset(save_path = "../data_cnn/train_%d-%d"%(1, 2), seed = [1, 2], batch_size = 4, augs = True, save_cache = False)
    #dataset = FloodDatasetCNN(seed = [1, 2], simulation_steps = 120, input_frames = 2, temporal_resolution = 2, bd = 1, augs = True)
    dataset = FloodDatasetCNN(seed = [1, 2], simulation_steps = 120, input_frames = 2, temporal_resolution = 2, bd = 1, augs = False, scene = 1)
    a = dataset[0]
    a = dataset[8]
    r'''
    b = dataset[9]
    c = dataset[13]
    c = dataset[935]
    c = dataset[936]
    '''
