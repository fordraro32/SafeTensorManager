import torch
import torch.distributed as dist
import os

def setup_distributed():
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend='nccl', world_size=world_size, rank=0)
            torch.cuda.set_device(0)
        else:
            print("Only one GPU available, not setting up distributed environment.")
    else:
        print("No GPUs available, running on CPU.")

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
