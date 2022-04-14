grid run \
search.py \
--lr "[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]" \
--batch_size "[64, 128, 256]" \
--epoch_size 20
parser = argparse.ArgumentParser("CNN_DARTS")#####
parser.add_argument('-seq', '--seq_length', type=int, default=30)
parser.add_argument('-step', '--step_length', type=int, default=1)
parser.add_argument('--unrolled', type=bool, default=False)
parser.add_argument('-aux', '--auxiliary', type= bool, default=False)
parser.add_argument('-aux_w', '--auxiliary_weight', type=float, default=0.1)
parser.add_argument('--drop_path_prob', type=float, default=0.2)
parser.add_argument('--init_C', type=int, default=16)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.025) ##smaller to avoid eror?.025
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0003)
parser.add_argument('--grad_clip', type=int, default=5)
parser.add_argument('-bs', '--batch_size', type=int, default=16) ###################배치사이즈
parser.add_argument('-sep', '--search_epoch_size', type=int, default=20)
parser.add_argument('-ep', '--epoch_size', type=int, default=50) ##약칭
#parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')#추가
parser.add_argument('--exp_path', type=Path, default=root_dir +'outputs/')


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python search.py -bs 32

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python train.py