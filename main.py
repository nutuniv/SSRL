from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from model import Model
from dataset import Dataset,Multi_Dataset
from train import train
from test import test
import option
from tqdm import tqdm
from utils import *
from config import *
import misc
from misc import *
import samplers


def dir_prepare(args):
    param_str = get_timestamp()
    logger_dir = args.save_dir+'logs/'+args.dataset+'/'+ param_str + '/'
    ckpt_dir = args.save_dir+'ckpts/'+args.dataset+'/' + param_str + '/'
    mkdir(logger_dir)
    mkdir(ckpt_dir)
    logger_path=logger_dir+args.model_name+'{}.log'.format(param_str)
    logger=get_logger(logger_path)
    logger.info('Train this model at time {}'.format(get_timestamp()))
    log_param(logger, args)

    return logger,param_str,ckpt_dir


if __name__ == '__main__':
    args = option.parser.parse_args()
    misc.init_distributed_mode(args)
    print("git:\n  {}\n".format(misc.get_sha()))
    
    config = Config(args)
    set_seeds(args.seed)
    logger, param_str, ckpt_dir = dir_prepare(args)

    model = Model(args, args.feature_size, args.batch_size, args.num_segments)

    for name, value in model.named_parameters():
        if value.requires_grad:
            logger.info(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print('load checkpoint')
    model = model.cuda()
    if args.train_part:
        param_dict = [
                {
                    "params":[p for n,p in model.named_parameters() if "patch_to_clip" in n or "linear_out" in n or 'clip_to_clip' in n and p.requires_grad],
                    "lr": config.lr[0]
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            "Aggregate_patch" in n  and p.requires_grad],
                    # "lr": config.lr[0]*0.1
                    "lr": config.lr[0]
                }
            ]
        optimizer = optim.Adam(param_dict,lr=config.lr[0], weight_decay=0.005)
    else:
        optimizer = optim.Adam(model.parameters(),
                             lr=config.lr[0], weight_decay=0.005)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    Dataset = Multi_Dataset if args.multi_patch_mode else Dataset

    norm_dataset = Dataset(args,test_mode=False,is_normal=True)
    abnorm_dataset = Dataset(args,test_mode=False,is_normal=False)
    test_dataset = Dataset(args,test_mode=True)
    if args.distributed:
        norm_sampler = samplers.DistributedSampler(norm_dataset)
        abnorm_sampler = samplers.DistributedSampler(abnorm_dataset)
        test_sampler = samplers.DistributedSampler(test_dataset,shuffle=False)
        train_nloader = DataLoader(norm_dataset, batch_size=args.batch_size, sampler=norm_sampler,
                                        num_workers=args.workers, 
                                        drop_last=True,pin_memory=False )
        train_aloader = DataLoader(abnorm_dataset, batch_size=args.batch_size,
                                        sampler=abnorm_sampler,
                                        num_workers=args.workers, 
                                        drop_last=True, pin_memory=False)

        test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=args.workers,
                                         drop_last=False, pin_memory=False)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    auc = test(test_loader, model, args, device,logger)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)
            norm_sampler.set_epoch(step)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)
            abnorm_sampler.set_epoch(step)

        train(args,loadern_iter, loadera_iter, model, args.batch_size, optimizer,  device)

        test_step = 5
        if step % test_step == 0 and step > 50:
            synchronize()

            auc = test(test_loader, model, args, device,logger)
            if not is_main_process():
                continue
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.module.state_dict(),ckpt_dir + args.model_name + param_str + '_step{}.pkl'.format(step))
                torch.save(model.module.state_dict(),ckpt_dir + 'best_auc.pkl')
                logger.info('epoch:{} auc\t{:.4f}'.format( test_info["epoch"][-1], test_info["test_AUC"][-1]))
    logger.info('best_auc\t{:.4f}'.format( best_AUC))

