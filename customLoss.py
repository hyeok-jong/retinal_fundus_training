
import torch

def custom_loss(output, labels, verbose = False):
    loss_function_label = torch.nn.BCEWithLogitsLoss()
    loss_function_age = torch.nn.MSELoss()
    loss_function_sex = torch.nn.BCEWithLogitsLoss()

    '''
    output and label shape : batch X 3
    '''
    loss_label = loss_function_label(output[:, 0], labels[:, 0])
    loss_age = loss_function_age(output[:, 1], labels[:, 1])
    loss_sex = loss_function_sex(output[:, 2], labels[:, 2])
    loss_total = loss_label + loss_age + loss_sex

    if not verbose:
        return loss_total
    elif verbose:
        return {
            'loss_label' : loss_label,
            'loss_age' : loss_age,
            'loss_sex' : loss_sex,
            'loss_total': loss_total
        }    