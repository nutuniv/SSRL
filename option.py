import argparse

parser = argparse.ArgumentParser(description='SSRL')
parser.add_argument('--local_rank', type = int, default = 0)
parser.add_argument('--data_dir', type = str, default = './dataset/' )
parser.add_argument('--feature_size', type = int, default = 2048, help = 'size of feature (default: 2048)')
parser.add_argument('--lr', type = str, default='[0.001]*150000', help = 'learning rates for steps(list form)')
parser.add_argument('--batch_size', type = int, default = 4, help = 'number of instances in a batch of data')
parser.add_argument('--workers', default = 2,type = int, help = 'number of workers in dataloader')
parser.add_argument('--model_name', type = str,default = 'ssrl', help = 'name to save model')
parser.add_argument('--pretrained_ckpt', default = None, help = 'ckpt for pretrained model')
parser.add_argument('--dataset', default = 'shanghai', help = 'dataset to train on')
parser.add_argument('--max_epoch', type = int, default = 150000, help = 'maximum iteration to train (default: 150000)')
parser.add_argument('--seed', type = int, default = 10, help = 'seeds')
parser.add_argument('--num_segments',type = int, default = 32, help = 'number of segments per video')
parser.add_argument('--checkpoint', type = str, default = None, help = 'checkpoint file')
parser.add_argument('--save_dir',type = str, default = 'output/', help = 'dir to save ckpt and logs')
parser.add_argument('--patch_mode', dest = 'patch_mode', action = 'store_true')
parser.add_argument('--train_part', dest = 'train_part', action = 'store_true')
parser.add_argument('--multi_patch_mode', dest = 'multi_patch_mode', action = 'store_true')
parser.add_argument('--multi_patch_size', type = int, nargs = '+', default = [23,35,47])
parser.set_defaults(train_part = False)
parser.set_defaults(patch_mode = False)
parser.set_defaults(multi_patch_mode = False)



