import datetime
import os
import time
import math
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection

from DeepDataMiningLearning.detection import utils
from DeepDataMiningLearning.detection.trainutils import create_aspect_ratio_groups, GroupedBatchSampler
from DeepDataMiningLearning.detection.dataset import get_dataset
from DeepDataMiningLearning.detection.myevaluator import simplemodelevaluate, modelevaluate

# Import your custom backbone
from hw1.backbone import get_efficientnet_backbone, CustomBackboneWithFPN

try:
    from torchinfo import summary
except ImportError:
    print("[INFO] Couldn't find torchinfo... installing it.")  # pip install -q torchinfo

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("--data-path", default="/data/cmpe258-sp24/013978029/dataset/Kitti", type=str, help="dataset path")
    parser.add_argument("--annotationfile", default="", type=str, help="dataset annotation file path")
    parser.add_argument("--dataset", default="kitti", type=str, help="dataset name")
    parser.add_argument("--model", default="custom_fpn", type=str, help="model name")
    parser.add_argument("--trainable", default=0, type=int, help="number of trainable layers (sequence) of backbone")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu, default: cuda)")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=60, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--saveeveryepoch", default=4, type=int, metavar="N", help="number of epochs to save")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.02, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)")
    parser.add_argument("--lr-steps", default=[16, 22], nargs="+", type=int, help="epochs to decrease lr")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="factor to decrease lr")
    parser.add_argument("--print-freq", default=5, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=-1, type=int, help="group sampler factor")
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold")
    parser.add_argument("--data-augmentation", default="hflip", type=str, help="data augmentation policy")
    parser.add_argument("--sync-bn", action="store_true", help="use synchronized batch norm")
    parser.add_argument("--test-only", action="store_true", help="only test the model")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="use deterministic algorithms only")
    parser.add_argument("--multigpu", default=False, type=bool, help="disable torch ddp")
    parser.add_argument("--world-size", default=4, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="backbone weights enum name to load")
    parser.add_argument("--amp", action="store_true", help="use torch.cuda.amp for mixed precision training")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor")
    parser.add_argument("--use-v2", action="store_true", help="use V2 transforms")
    parser.add_argument("--expname", default="experiment", help="experiment name")
    parser.add_argument("--max-batches", default=None, type=int, help="maximum number of batches to process for testing")

    return parser

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.expname)
        utils.mkdir(args.output_dir)

    if args.multigpu:
        utils.init_distributed_mode(args)
        args.distributed = True
    else:
        args.distributed = False
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    print("Loading data")

    if args.dataset.lower() == "kitti":
        # Use KITTI-specific dataset and transformations
        from DeepDataMiningLearning.detection.dataset_kitti import KittiDataset, get_transformsimple

        transform_train = get_transformsimple(is_train=True)
        transform_val = get_transformsimple(is_train=False)
        dataset = KittiDataset(root=args.data_path, train=True, split='train', transform=transform_train)
        dataset_test = KittiDataset(root=args.data_path, train=False, split='val', transform=transform_val)
        num_classes = dataset.numclass
    else:
        # Use get_dataset for other datasets
        dataset, num_classes = get_dataset(args.dataset, is_train=True, is_val=False, args=args)
        dataset_test, _ = get_dataset(args.dataset, is_train=False, is_val=True, args=args)

    print("train set len:", len(dataset))
    print("test set len:", len(dataset_test))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=1, collate_fn=utils.collate_fn
    )

    print("Creating model")
    backbone = get_efficientnet_backbone(pretrained=True)
    layer_names = ["0.1.0.block", "0.2.1.block", "0.4.2.block", "0.6.3.block"]
    model = CustomBackboneWithFPN(backbone, layer_names, out_channels=256).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        simplemodelevaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler, max_batches=args.max_batches)
        lr_scheduler.step()
        if (epoch + 1) % args.saveeveryepoch == 0 or epoch == args.epochs:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch + 1}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
            modelevaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, max_batches=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    batch_count = 0  # Counter to track the number of processed batches

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        if max_batches is not None and batch_count >= max_batches:
            break  # Stop processing more batches if the limit is reached

        # Unpack images and targets from the batch
        images, targets = batch

        if isinstance(images, tuple):
            images = list(images)  # Ensure images is a list of tensors

        # Debugging: Log image sizes to ensure consistency
        image_sizes = [image.size() for image in images]
        print(f"Image sizes in this batch: {image_sizes}")
        consistent_size = all(size == image_sizes[0] for size in image_sizes)
        if not consistent_size:
            print(f"Warning: Inconsistent image sizes detected in batch {batch_count + 1}: {image_sizes}")

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            # Forward pass and handle output format
            output = model(images, targets) if targets is not None else model(images)
            
            # Handle tuple output from the model
            if isinstance(output, tuple):
                loss_dict = output[0]  # Assuming the first element is the loss dictionary
            else:
                loss_dict = output

            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())

        # Reduce and accumulate loss for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_count += 1  # Increment the batch counter

    return metric_logger

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
