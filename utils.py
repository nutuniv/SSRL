import torch.nn
import numpy as np
import logging
import random
import os
import torch
import time

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret



class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.vals=[]

    def __format__(self, format_spec):
        f=0
        if len(self.vals)!=0:
            f=(sum(self.vals)/len(self.vals))
        return ('{:'+format_spec+'}').format(f)

    def val(self):
        if len(self.vals) != 0:
            f = sum(self.vals) / len(self.vals)
        else:
            f=0
        return f

    def update(self,val):
        if isinstance(val,np.ndarray):
            self.vals.append(val[0])
        elif isinstance(val,np.float64):
            self.vals.append(val)
        else:
            self.vals.append(val.detach().cpu().item())

def show_params(args):
    params=vars(args)
    keys=sorted(params.keys())

    for k in keys:
        print(k,'\t',params[k])

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def log_param(logger,args):
    params=vars(args)
    keys=sorted(params.keys())
    for k in keys:
        logger.info('{}\t{}'.format(k,params[k]))

def get_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def mkdir(dir):
    if not os.path.exists(dir):
        try:os.makedirs(dir)
        except:pass

def set_seeds(seed):
    print('set seed {}'.format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_epoch_idx(epoch,milestones):
    count=0
    for milestone in milestones:
        if epoch>milestone:
            count+=1
    return count


def weights_normal_init(model, dev=0.01):
    import torch
    from torch import nn

    # torch.manual_seed(2020)
    # torch.cuda.manual_seed_all(2020)
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias!=None:
                    m.bias.data.fill_(0)