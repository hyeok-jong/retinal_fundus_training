import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


def accuracy(output, label, threshold = 0.5):
    output = output.cpu().detach().squeeze()
    label = label.cpu().detach()
    
    prediction = (output > threshold)*1
    corrects = (label == prediction).sum()
    
    return corrects / len(label)
    


def auroc(output, label, save_fig, save_dir):
    
    output = output.cpu().detach().squeeze().numpy()
    label = label.cpu().detach().numpy()
    score = roc_auc_score(label, output)
    
    if save_fig:
        FIG = plt.figure(figsize = (10, 9))
        fontsize = 20
        fpr, tpr, _ = roc_curve(label, output)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize = fontsize)
        plt.ylabel('True Positive Rate', fontsize = fontsize)
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize = fontsize)
        plt.legend(loc="lower right", fontsize = fontsize)
        FIG.tight_layout()
        # FIG.savefig(save_dir.replace('.png', '_auroc.png'), dpi = 200)
        FIG.savefig(save_dir.replace('.png', '_auroc.svg'))
        plt.close()
    
    return score

def auprc(output, label, save_fig, save_dir):
    
    output = output.cpu().detach().squeeze().numpy()
    label = label.cpu().detach().numpy()
    score = average_precision_score(label, output)
    
    if save_fig:
        FIG = plt.figure(figsize = (10, 9))
        fontsize = 20
        precision, recall, _ = precision_recall_curve(label, output)
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {score:.2f})')
        plt.xlabel('Recall', fontsize = fontsize)
        plt.ylabel('Precision', fontsize = fontsize)
        plt.title('Precision-Recall Curve', fontsize = fontsize)
        plt.legend(loc="lower left", fontsize = fontsize)
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        FIG.tight_layout()
        # FIG.savefig(save_dir.replace('.png', '_auprc.png'), dpi = 200)
        FIG.savefig(save_dir.replace('.png', '_auprc.svg'))
        plt.close()
    
    return score

def compute_metrics(output, label, save_fig = False, save_dir = None, prefix = ''):
    return {
        f'{prefix}acc': accuracy(output, label),
        f'{prefix}auroc' : auroc(output, label, save_fig, save_dir),
        f'{prefix}auprc' : auprc(output, label, save_fig, save_dir)
    }
    

def get_ddp_results(result_dict, prefix, rank, world_size):
    return_dict = dict()
    for key, val in result_dict.items():
        if isinstance(val, torch.Tensor):
            metric_tensor = val.clone().detach().to(dtype=torch.float32, device=rank)
        else:
            metric_tensor = torch.tensor(val, dtype=torch.float32).to(device=rank)
        return_list = [torch.zeros(1, device = rank, dtype = torch.float32) for _ in range(world_size)]
        torch.distributed.all_gather(return_list, metric_tensor)
        metric = sum(return_list)/len(return_list)
        return_dict[f'{prefix}{key}'] = metric.cpu().numpy()
        
    return return_dict
    
    
@torch.no_grad()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
    
def set_random(random_seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)





# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal

def color_print(text, color='red', bg_color=None, bold=False, underline=False, end = None):
    # make all types to string.
    text = f'{text}'
    color_dict = {
        'black': '\033[30m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'orange': '\033[33m',
        'purple': '\033[35m',
        'pink': '\033[95m',
        'white': '\033[97m'
    }

    bg_color_dict = {
        'black': '\033[40m',
        'red': '\033[41m',
        'green': '\033[42m',
        'yellow': '\033[43m',
        'blue': '\033[44m',
        'magenta': '\033[45m',
        'cyan': '\033[46m',
        'white': '\033[47m',
    }

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    STRIKETHROUGH = '\033[09m'
    ENDC = '\033[0m'

    if bold:
        text = BOLD + text
    if underline:
        text = UNDERLINE + text
    if color:
        text = color_dict[color.lower()] + text
    if bg_color:
        text = bg_color_dict[bg_color.lower()] + text
    print(text + ENDC, end = end)
    
    
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
