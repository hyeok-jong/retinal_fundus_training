



import argparse
from datetime import datetime
import pytz

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def set_parser():
    parser = argparse.ArgumentParser('')

    parser.add_argument('--master_port', type = str, default = '12357')
    
    parser.add_argument('--model_name', type = str)

    parser.add_argument('--learning_method', type = str)

    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--lr', type = float)
    parser.add_argument('--epochs', type = int)
    parser.add_argument('--stop_epoch', type = int, default = False,
    help = 'stop epoch for dubugging or hyperparameter tuning. (for grid search runs only few epochs)')

    parser.add_argument('--optimizer', type = str)
    parser.add_argument('--scheduler', type = str)
    parser.add_argument('--weight_decay', type = float)
    
    parser.add_argument('--image_size', type = int)
    
    parser.add_argument('--mixup', type = str2bool)

    parser.add_argument('--agesex', type = str2bool)
    
    args = parser.parse_args()

    exper_name = ''
    for key, val in vars(args).items():
        exper_name += f'{key}:{val}__'

    korea_timezone = pytz.timezone('Asia/Seoul')
    korea_time = datetime.now(korea_timezone)
    exper_name += korea_time.strftime('%Y-%m-%d %H:%M:%S').replace(' ', '_')
    
    args.exper_name = exper_name
    
    return args