# Basics
import os
import numpy as np
import argparse as ap
import datetime as dt

# mlperf logger
import utils.mlperf_log_utils as mll

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils import mkldnn as mkldnn_utils
from torch.backends import mkldnn

# Custom
#from utils import utils
from utils import losses
from utils import parsing_helpers as ph
#from data import cam_hdf5_dataset as cam
from architecture import deeplab_xception

#warmup scheduler
have_warmup_scheduler = False

#comm wrapper
from utils import comm

#dict helper for argparse
class StoreDictKeyPair(ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


#main function
def main(pargs):

    # set up logging
    pargs.logging_frequency = max([pargs.logging_frequency, 1])
    log_file = os.path.normpath(os.path.join(pargs.output_dir, "logs", pargs.run_tag + ".log"))
    logger = mll.mlperf_logger(log_file, "deepcam", "Umbrella Corp.")
    logger.log_start(key = "init_start", sync = True)        
    
    #set seed
    seed = 333
    logger.log_event(key = "seed", value = seed)
    
    # Some setup
    torch.manual_seed(seed)
    device = torch.device("cpu")

    comm_size = 1
    comm_rank = 0
    # initial logging
    logger.log_event(key = "global_batch_size", value = (pargs.local_batch_size * comm_size))
    logger.log_event(key = "optimizer", value = pargs.optimizer)

    # Define architecture
    n_input_channels = len(pargs.channels)
    n_output_channels = 3
    net = deeplab_xception.DeepLabv3_plus(n_input = n_input_channels, 
                                          n_classes = n_output_channels, 
                                          os=16, pretrained=False, 
                                          rank = comm_rank)
    net.to(device)
    #print(net)
    #mkldnn_utils.to_mkldnn(net)
    #print(net)

    #select loss
    loss_pow = pargs.loss_weight_pow
    #some magic numbers
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    fpw_1 = 2.61461122397522257612
    fpw_2 = 1.71641974795896018744
    criterion = losses.fp_loss

    #select optimizer
    optimizer = None
    if pargs.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    else:
        raise NotImplementedError("Error, optimizer {} not supported".format(pargs.optimizer))

    start_step = 0
    start_epoch = 0
        
    #select scheduler
    if pargs.lr_schedule:
        scheduler_after = ph.get_lr_schedule(pargs.start_lr, pargs.lr_schedule, optimizer, last_step = start_step)

        if have_warmup_scheduler and (pargs.lr_warmup_steps > 0):
            scheduler = GradualWarmupScheduler(optimizer, multiplier=pargs.lr_warmup_factor, total_epoch=pargs.lr_warmup_steps, after_scheduler=scheduler_after)
        else:
            scheduler = scheduler_after
        
    #broadcast model and optimizer state
    steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False).to(device)

    #unpack the bcasted tensor
    start_step = steptens.cpu().numpy()[0]
    start_epoch = steptens.cpu().numpy()[1]

    # dummy data
    inputs = torch.randn(2, 16, 768, 1152)
    label = torch.ones([2, 768, 1152])
    inputs_val = torch.randn(2, 16, 768, 1152)
    label_val = torch.ones([2, 768, 1152])

    inputs = inputs.to_mkldnn()

    step = start_step
    epoch = start_epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    net.train()

    # start trining
    logger.log_end(key = "init_stop", sync = True)
    logger.log_start(key = "run_start", sync = True)

    # training loop
    with torch.autograd.profiler.profile() as prof:
        while True:

            # start epoch
            logger.log_start(key = "epoch_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=True)

            # epoch loop
            #for inputs, label, filename in train_loader:
            for i in range(10):
                # send to device
                inputs = inputs.to(device)
                label = label.to(device)
                
                # forward pass
                outputs = net.forward(inputs)
                
                # Compute loss and average across nodes
                loss = criterion(outputs.to_dense(), label, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # step counter
                step += 1
                
                if pargs.lr_schedule:
                    current_lr = scheduler.get_last_lr()[0]
                    scheduler.step()

                #log if requested
                if (step % pargs.logging_frequency == 0):

                    # allreduce for loss
                    loss_avg = loss.detach()
                    loss_avg_train = loss_avg.item() / float(comm_size)

                    logger.log_event(key = "learning_rate", value = current_lr, metadata = {'epoch_num': epoch+1, 'step_num': step})
                    logger.log_event(key = "train_loss", value = loss_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
                
                # validation step if desired
                if (step % pargs.validation_frequency == 0):
                    
                    #eval
                    net.eval()
                    
                    count_sum_val = torch.Tensor([0.]).to(device)
                    loss_sum_val = torch.Tensor([0.]).to(device)
                    iou_sum_val = torch.Tensor([0.]).to(device)
                    
                    # disable gradients
                    with torch.no_grad():
                    
                        # iterate over validation sample
                        step_val = 0
                        # only print once per eval at most
                        visualized = False
                        #for inputs_val, label_val, filename_val in validation_loader:
                        for j in range(10):
                            
                            #send to device
                            inputs_val = inputs_val.to(device)
                            label_val = label_val.to(device)
                            
                            # forward pass
                            outputs_val = net.forward(inputs_val)

                            # Compute loss and average across nodes
                            loss_val = criterion(outputs_val, label_val, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
                            loss_sum_val += loss_val
                            
                            #increase counter
                            count_sum_val += 1.
                            
                            #increase eval step counter
                            step_val += 1
                            
                            if (pargs.max_validation_steps is not None) and step_val > pargs.max_validation_steps:
                                break
                            
                    loss_avg_val = loss_sum_val.item() / count_sum_val.item()
                    
                    # print results
                    logger.log_event(key = "eval_loss", value = loss_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})

                    # set to train
                    net.train()
                
                #save model if desired
                if (pargs.save_frequency > 0) and (step % pargs.save_frequency == 0):
                    logger.log_start(key = "save_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
                    if comm_rank == 0:
                        checkpoint = {
                            'step': step,
                            'epoch': epoch,
                            'model': net.state_dict(),
                            'optimizer': optimizer.state_dict()
                        }
                        torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_step_" + str(step) + ".cpt") )
                    logger.log_end(key = "save_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
                
            # log the epoch
            logger.log_end(key = "epoch_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
            epoch += 1
            
            # are we done?
            if epoch >= pargs.max_epochs:
                break

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    # run done
    logger.log_end(key = "run_stop", sync = True)
    

if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--wireup_method", type=str, default="nccl-openmpi", choices=["nccl-openmpi", "nccl-slurm", "nccl-slurm-pmi", "mpi"], help="Specify what is used for wiring up the ranks")
    AP.add_argument("--wandb_certdir", type=str, default="/opt/certs", help="Directory in which to find the certificate for wandb logging.")
    AP.add_argument("--run_tag", type=str, help="Unique run tag, to allow for better identification")
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable from rank 0")
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs to train")
    AP.add_argument("--save_frequency", type=int, default=100, help="Frequency with which the model is saved in number of steps")
    AP.add_argument("--validation_frequency", type=int, default=100, help="Frequency with which the model is validated")
    AP.add_argument("--max_validation_steps", type=int, default=None, help="Number of validation steps to perform. Helps when validation takes a long time. WARNING: setting this argument invalidates submission. It should only be used for exploration, the final submission needs to have it disabled.")
    AP.add_argument("--logging_frequency", type=int, default=100, help="Frequency with which the training progress is logged. If not positive, logging will be disabled")
    AP.add_argument("--training_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during training")
    AP.add_argument("--validation_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during validation")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], help="Channels used in input")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "LAMB"], help="Optimizer to use (LAMB requires APEX support).")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_weight_pow", type=float, default=-0.125, help="Decay factor to adjust the weights")
    AP.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for linear LR warmup")
    AP.add_argument("--lr_warmup_factor", type=float, default=1., help="Multiplier for linear LR warmup")
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--target_iou", type=float, default=0.82, help="Target IoU score.")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--amp_opt_level", type=str, default="O0", help="AMP optimization level")
    AP.add_argument("--enable_wandb", action='store_true')
    AP.add_argument("--resume_logging", action='store_true')
    pargs = AP.parse_args()
    
    #run the stuff
    main(pargs)
