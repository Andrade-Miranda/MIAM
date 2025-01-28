import torch
import torch.distributed as dist
import random
import numpy as np
import logging

# Utility Functions
def set_global_seed(seed):
    """Set the global seed for reproducibility."""
    logging.info(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_distributed_mode(args):
    """Initialize distributed training if enabled."""
    if args.distributed:
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank
        )
        torch.cuda.set_device(args.local_rank)
        logging.info(f"Distributed training initialized: rank {args.local_rank}/{args.world_size}")
    else:
        logging.info("Distributed training not enabled.")

