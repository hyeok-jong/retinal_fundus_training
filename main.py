

import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import wandb

from tqdm import tqdm

from datasets import set_datasets
from ddp_functions import setup, cleanup, set_dataloader
from train_functions import train, inference, set_optimizer, set_lr_scheduler
from utils import get_ddp_results, get_lr, color_print
from CAWR import CosineAnnealingWarmUpRestarts
from models import set_model
from args import set_parser
from utils import compute_metrics, set_random

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from datetime import datetime
from pytz import timezone



def ddp_main(rank, world_size, train_dataset, valid_dataset, test_dataset, args_parser, verbose = True):
    # 이 함수가 world_size개수 만큼 parallel하게 실행 된다고 보면 된다.
    # 즉, ddp_main 함수는 gpu 하나의 실행 방식이다.
    setup(rank, world_size, args_parser.master_port)
    if rank == 0:
        wandb.init(project = 'retinal_fundus_1101', name = args_parser.exper_name)
        
        # 이거 해주면 wanbd에서 add panel -> paramter importance 추가해서 어떤 hyperparameter가 중요한지 볼 수 있음.
        for key, value in vars(args_parser).items():
            setattr(wandb.config, key, value)
            print(key, value)
    

    batch_size = args_parser.batch_size
    num_workers = 4

    train_loader, train_sampler = set_dataloader(
        dataset = train_dataset,
        world_size = world_size,
        rank = rank,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = True,
        return_sampler = True
    )

    valid_loader = set_dataloader(
        dataset = valid_dataset,
        world_size = world_size,
        rank = rank,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = True
    )

    test_loader = set_dataloader(
        dataset = test_dataset,
        world_size = world_size,
        rank = rank,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = True
    )
    

    model = set_model(args_parser.model_name, rank, learning_method = args_parser.learning_method, 
        out_features = 3 if args_parser.agesex == True else 1)

    

    # 어떤 부분인 update되는지 monitoring 하기 위해
    model.train()
    if rank == 0:
        init_model = dict()
        for k, v in model.state_dict().items():
            init_model[f'module.{k}'] = v.detach().cpu()


    # DDP는 엄밀하게는 gpu별로 모델을 뿌려주는것이 아니다.
    # gpu로 뿌려주는건 mp.spawn에서 multi로 실행될때 각 rank마다 model load를 하는것이다.
    # DDP는 여러개의 gpu에 있는 모델의 역전파 연산을 한번에 모으기 위해서만 필요한 wrapper이다. 
    # 따라서 requires_grad가 모두 False인 경우 DDP로 하면 안된다. (오류 남.)
    # if args_parser.classifier_only or args_parser.partial_finetune:   
    #     model = model.to(rank)
    # else:
    model = DDP(model, device_ids=[rank])


    # BCEWithLogitsLoss는 sigmoid 포함한거고 여기서는 manually 추가했기 때문에 BCELoss를 사용한다.
    # output = torch.tensor([.1, .4])
    # label = torch.tensor([1., 0.])
    # loss1 = torch.nn.BCELoss()
    # loss2 = torch.nn.BCEWithLogitsLoss()
    # print(loss1(torch.sigmoid(output), label))
    # print(loss2(output, label))
    
    loss_function = torch.nn.BCELoss().cuda(rank)
    if args_parser.agesex:
        from customLoss import custom_loss
        loss_function = custom_loss
    optimizer = set_optimizer(args_parser.optimizer, list(model.parameters()), 1e-6 if args_parser.scheduler == 'CAWR' else args_parser.lr, weight_decay = args_parser.weight_decay)
    

    lr_scheduler = set_lr_scheduler(
        optimizer = optimizer,
        epochs = args_parser.epochs,
        learning_rate = args_parser.lr,
        name = args_parser.scheduler
    )

    best_valid_auroc = -1.

    # 1부터 시작하는게 좋은데, wanbd가 0부터 시작해서 맞출려고 이렇게함.
    if args_parser.stop_epoch:
        final_epoch = args_parser.stop_epoch
    else:
        final_epoch = args_parser.epochs
        
    for epoch in tqdm(range(0, final_epoch)):

        # see ddp_functions.py
        dist.barrier()

        # train sampler를 조정해서 각 gpu에 들어가는 것 차체를 shuffle
        train_sampler.set_epoch(epoch)
        
        train_output, train_labels = train(train_loader, model, loss_function, optimizer, rank, epoch, args_parser)
        valid_output, valid_labels = inference(valid_loader, model, loss_function, rank, epoch, args_parser)
        test_output, test_labels = inference(test_loader, model, loss_function, rank, epoch, args_parser)

        if rank == 0:

            train_results = compute_metrics(train_output[:, 0], train_labels[:, 0], prefix = 'train_', save_fig = True, save_dir = f"./experiments/{args_parser.exper_name}/figs/train_{epoch}.png")
            valid_results = compute_metrics(valid_output[:, 0], valid_labels[:, 0], prefix = 'valid_', save_fig = True, save_dir = f"./experiments/{args_parser.exper_name}/figs/valid_{epoch}.png")
            test_results = compute_metrics(test_output[:, 0], test_labels[:, 0], prefix = 'test_', save_fig = True, save_dir = f"./experiments/{args_parser.exper_name}/figs/test_{epoch}.png")

            results = dict()
            for i in [train_results, valid_results, test_results]:
                results.update(i)
            
            train_loss = loss_function(train_output.cuda(), train_labels.cuda())
            valid_loss = loss_function(valid_output.cuda(), valid_labels.cuda())
            test_loss = loss_function(test_output.cuda(), test_labels.cuda())

            results.update({'train_loss_label' : train_loss['loss_label'].detach().cpu().item()})
            results.update({'train_loss_age' : train_loss['loss_age'].detach().cpu().item()})
            results.update({'train_loss_sex' : train_loss['loss_sex'].detach().cpu().item()})
            results.update({'train_loss_total' : train_loss['loss_total'].detach().cpu().item()})

            results.update({'valid_loss_label' : valid_loss['loss_label'].detach().cpu().item()})
            results.update({'valid_loss_age' : valid_loss['loss_age'].detach().cpu().item()})
            results.update({'valid_loss_sex' : valid_loss['loss_sex'].detach().cpu().item()})
            results.update({'valid_loss_total' : valid_loss['loss_total'].detach().cpu().item()})

            results.update({'test_loss_label' : test_loss['loss_label'].detach().cpu().item()})
            results.update({'test_loss_age' : test_loss['loss_age'].detach().cpu().item()})
            results.update({'test_loss_sex' : test_loss['loss_sex'].detach().cpu().item()})
            results.update({'test_loss_total' : test_loss['loss_total'].detach().cpu().item()})

            results.update({'learning_rate' : get_lr(optimizer)})
            
            color_print(f"Current time : {datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')}", 'yellow', bold = True)
            print(f"Epoch {epoch}:")
            for key, val in results.items():
                print(key, val)
            
            save_dict = {'model_state_dict' : model.state_dict()}
            save_dict.update(results)
            save_dict.update({'epoch' : epoch})
            save_dict.update({'args' : args_parser})
            
            # save all epochs
            # torch.save(save_dict, f"./experiments/{args_parser.exper_name}/saves/model_epoch_{epoch}.pth")
            # print(f"Model saved for epoch {epoch}")

            current_valid_auroc = results['valid_auroc']
            if current_valid_auroc >= best_valid_auroc:
                torch.save(save_dict, f"./experiments/{args_parser.exper_name}/saves/model_best.pth")
            
            wandb.log(results)

            # 모델이 특정 부분만 update되는지 추가 확인
            if (verbose) and (epoch % 10 == 0):
                init_keys = init_model.keys()
                current_keys = model.state_dict().keys()

                if init_keys != current_keys:
                    raise ValueError(f"Key mismatch: {init_keys} != {current_keys}")
        
                for (k0, v0), (k1, v1) in zip(init_model.items(),  model.state_dict().items()):
                    v0_gpu = v0.to(rank)
                    v1_gpu = v1 if v1.is_cuda else v1.to(rank)
                    if torch.equal(v0_gpu, v1_gpu):
                        color_print(f'{k0} \U0001F976 freezed', 'blue', end = ' difference : ')
                    else:
                        color_print(f'{k0} \U0001F525 updated', 'red', end = ' difference : ')
                    diff = (v0_gpu - v1_gpu).sum()
                    if diff != 0:
                        print(diff.item())

        lr_scheduler.step()

    if rank == 0:
        load = torch.load(f"./experiments/{args_parser.exper_name}/saves/model_best.pth")
        wandb.log({f'best_{k}':v for k, v in load.items() if 'test' in k or 'epoch' in k })
    
    cleanup()
    
    
if __name__ == '__main__':
    # set_random(0)

    args_parser = set_parser()
    os.makedirs(f'./experiments/{args_parser.exper_name}/saves', exist_ok = True)
    os.makedirs(f'./experiments/{args_parser.exper_name}/figs', exist_ok = True)
    world_size = torch.cuda.device_count()
    train_dataset, valid_dataset, test_dataset = set_datasets(args_parser.image_size)

    # world size 만큼 열어서 동시에 돌아가게 하는 역할
    try:
        torch.multiprocessing.spawn( 
            fn = ddp_main, 
            args = (world_size, train_dataset, valid_dataset, test_dataset, args_parser), 
            nprocs = world_size, 
            join = True
        )
    except:
        print('Interrupted')
        try: 
            dist.destroy_process_group()  
        except KeyboardInterrupt: 
            os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")

