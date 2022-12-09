import argparse
import os
import shutil
import time
import random
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from monai.config import KeysCollection
from monai.data import Dataset
from monai.data.image_reader import NumpyReader
from monai.metrics import Cumulative, CumulativeAverage
from monai.networks.nets import milmodel
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    ScaleIntensityRanged,
    ToTensord,
)
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, auc, precision_recall_curve
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
import torchvision.models.resnet
from thyroid_utils import *

from torchvision.models.resnet import resnet18, resnet34, resnet50

def train_epoch(model, loader, optimizer, scaler, epoch, args):
    """One train epoch over the dataset"""

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()
    PREDS = Cumulative()
    TARGETS = Cumulative()
    SCORE = Cumulative()


    start_time = time.time()
    loss, acc = 0.0, 0.0
    total = torch.tensor([], device=args.rank)

    for idx, batch_data in enumerate(loader):

        data, target = batch_data["image"], batch_data["label"].cuda(args.rank) 
        #data = data.permute((0,1,4,2,3)) 
        data = RandPatch(args.train_patch, data)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            logits = model(data.cuda(args.rank))
            loss = criterion(logits[0],target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        pred = logits>0
        target = target
        acc = (pred == target).float().mean()

        run_loss.append(loss)
        run_acc.append(acc)
        loss = run_loss.aggregate()
        acc = run_acc.aggregate()

        SCORE.extend(logits.sigmoid())
        PREDS.extend(pred)
        TARGETS.extend(target)

        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.epochs, idx, len(loader)),
                "loss: {:.4f}".format(loss),
                "acc: {:.4f}".format(acc),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()

    PREDS = PREDS.get_buffer().cpu().numpy()
    TARGETS = TARGETS.get_buffer().cpu().numpy()
    SCORE = SCORE.get_buffer().cpu().numpy()
    PRE, REC, _ = precision_recall_curve(TARGETS.astype(np.float64),SCORE.astype(np.float64))

    #tn, fp, fn, tp = confusion_matrix(TARGETS.astype(np.float64),PREDS.astype(np.float64)).ravel()
    f1 = f1_score(TARGETS.astype(np.float64),PREDS.astype(np.float64))
    auroc = roc_auc_score(TARGETS.astype(np.float64),SCORE.astype(np.float64))
    auprc = auc(REC, PRE)
    if epoch == args.epochs -1:
        np.save('train_Targets_fl',TARGETS.astype(np.float64))
        np.save('train_Preds_fl',PREDS.astype(np.float64))
        np.save('test_Scores_fl',SCORE.astype(np.float64))
    return loss, acc, f1, auroc, auprc

def val_epoch(model, loader, epoch, args):
    """One validation epoch over the dataset"""
    model.eval()

    model2 = model if not args.distributed else model.module
    calc_head = model2.calc_head

    criterion = nn.BCEWithLogitsLoss()
    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()
    PREDS = Cumulative()
    TARGETS = Cumulative()
    SCORE = Cumulative()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    with torch.no_grad():

        for idx, batch_data in enumerate(loader):

            #if max_tiles is not None and batch_data["label"].shape[1] > max_tiles :
            data, target = batch_data["image"], batch_data["label"].cuda(args.rank)
            data = data.permute((0,1,4,2,3)) 
            with autocast(enabled=args.amp):
                if data.shape[1] > args.max_tile:

                    logits = []

                    for i in range(int(np.ceil(data.shape[1] / float(args.max_tile)))):
                        data_slice = data[:, i * args.max_tile : (i + 1) * args.max_tile].cuda(args.rank)
                        logits_slice = model(data_slice, no_head=True)
                        logits.append(logits_slice)

                    logits = torch.cat(logits, dim=1)

                    logits = calc_head(logits)

                else:
                    # if number of instances is not big, we can run inference directly
                    logits = model(data.cuda(args.rank))

                loss = criterion(logits[0],target)

            pred = logits>0
            target = target
            acc = (pred == target).float().mean()

            run_loss.append(loss)
            run_acc.append(acc)
            loss = run_loss.aggregate()
            acc = run_acc.aggregate()

            SCORE.extend(logits.sigmoid())
            PREDS.extend(pred)
            TARGETS.extend(target)

            if args.rank == 0:
                print(
                    "Val epoch {}/{} {}/{}".format(epoch, args.epochs, idx, len(loader)),
                    "loss: {:.4f}".format(loss),
                    "acc: {:.4f}".format(acc),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    PREDS = PREDS.get_buffer().cpu().numpy()
    TARGETS = TARGETS.get_buffer().cpu().numpy()
    SCORE = SCORE.get_buffer().cpu().numpy()
    PRE, REC, _ = precision_recall_curve(TARGETS.astype(np.float64),SCORE.astype(np.float64))

    tn, fp, fn, tp = confusion_matrix(TARGETS.astype(np.float64),PREDS.astype(np.float64)).ravel()
    print(tn, fp, fn, tp)
    f1 = f1_score(TARGETS.astype(np.float64),PREDS.astype(np.float64))
    auroc = roc_auc_score(TARGETS.astype(np.float64),SCORE.astype(np.float64))
    auprc = auc(REC, PRE)
    if epoch == args.epochs -1:
        np.save('test_Targets_fl',TARGETS.astype(np.float64))
        np.save('test_Preds_fl',PREDS.astype(np.float64))
        np.save('test_Scores_fl',SCORE.astype(np.float64))
    return loss, acc, f1, auroc, auprc


#focal loss code from : https://velog.io/@heaseo/Focalloss-%EC%84%A4%EB%AA%85
# set alpha to 0 

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def save_checkpoint(model, epoch, args, filename="model_focalloss.pt", best_acc=0):
    """Save checkpoint"""

    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()

    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


class LabelEncodeIntegerGraded(MapTransform):
    """
    Convert an integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)

    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed. Defaults to ``'label'``.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        num_classes: int,
        keys: KeysCollection = "label",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            label = int(d[key])

            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.0
            # alternative oneliner lz=(np.arange(self.num_classes)<int(label)).astype(np.float32) #same oneliner
            d[key] = lz

        return d


def main_worker(gpu, args):
    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * torch.cuda.device_count() + gpu
        # dist.init_process_group(
        #     backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        # )
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,  world_size=args.world_size, rank=args.rank
        )

    print(args.rank, " gpu", args.gpu)

    torch.cuda.set_device(args.gpu)  # use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True


    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.epochs)

    training_list = datalist('train', oversample = 4)
    validation_list = datalist('test', oversample = 4)
    
    if args.quick:  # for debugging on a small subset
        training_list = training_list[:16]
        validation_list = validation_list[235:251] #라벨 섞여야 auroc 나옴
        args.epochs = 3

    transformD = Compose(
        [
            LoadImaged(keys=["image"], reader=NumpyReader, dtype=np.uint8, image_only=True),
            ScaleIntensityRanged(keys=["image"], a_min=np.float32(255), a_max=np.float32(0)),
            ToTensord(keys=["image", "label"])
        ]
    )

    dataset_train = Dataset(data=training_list, transform=transformD)
    dataset_valid = Dataset(data=validation_list, transform=transformD)

    train_sampler = DistributedSampler(dataset_train) if args.distributed else None
    val_sampler = DistributedSampler(dataset_valid, shuffle=False) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context="spawn",
        sampler=train_sampler,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context="spawn",
        sampler=val_sampler,
    )

    if args.rank == 0:
        print("Dataset training:", len(dataset_train), "validation:", len(dataset_valid))

    model = milmodel.MILModel(num_classes=args.num_classes, pretrained=True, mil_mode=args.mil_mode)

    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    if args.validate:
        # if we only want to validate existing checkpoint
        epoch_time = time.time()
        val_loss, val_acc, val_f1, val_auroc, val_auprc = val_epoch(model, valid_loader, epoch=0, args=args)
        if args.rank == 0:
            print(
                "***************Final validation",
                "loss: {:.4f}".format(val_loss),
                "acc: {:.4f}".format(val_acc),
                "f1 score: {:.4f}".format(val_f1),
                "AUROC: {:.4f}".format(val_auroc),
                "AUPRC: {:.4f}".format(val_auprc),
                "time {:.2f}s".format(time.time() - epoch_time)
            )
        exit(0)

    params = model.parameters()

    if args.mil_mode in ["att_trans", "att_trans_pyramid"]:
        m = model if not args.distributed else model.module
        params = [
            {"params": list(m.attention.parameters()) + list(m.myfc.parameters()) + list(m.net.parameters())},
            {"params": list(m.transformer.parameters()), "lr": 6e-6, "weight_decay": 0.1},
        ]

    optimizer = torch.optim.AdamW(params, lr=args.optim_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # if args.logdir is not None and args.rank == 0:
    #     writer = SummaryWriter(log_dir=args.logdir)
    #     if args.rank == 0:
    #         print("Writing Tensorboard logs to ", writer.log_dir)
    # else:
    #     writer = None

    ###RUN TRAINING
    n_epochs = args.epochs
    val_acc_max = 0.0

    scaler = None
    if args.amp:  # new native amp
        scaler = GradScaler()

    for epoch in range(start_epoch, n_epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
            torch.distributed.barrier()

        print(args.rank, time.ctime(), "Epoch:", epoch)

        epoch_time = time.time()
        train_loss, train_acc, train_f1, train_auroc, train_auprc = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args)

        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, n_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "acc: {:.4f}".format(train_acc),
                "f1 score: {:.4f}".format(train_f1),
                "AUROC: {:.4f}".format(train_auroc),
                "AUPRC: {:.4f}".format(train_auprc),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

        # if args.rank == 0 and writer is not None:
        #     writer.add_scalar("train_loss", train_loss, epoch)
        #     writer.add_scalar("train_acc", train_acc, epoch)

        if args.distributed:
            torch.distributed.barrier()

        b_new_best = False
        val_acc = 0
        # run validation only once
        if epoch == n_epochs-1:

            epoch_time = time.time()
            val_loss, val_acc, val_f1, val_auroc, val_auprc = val_epoch(model, valid_loader, epoch=epoch, args=args)
            if args.rank == 0:
                print(
                    "***************Final validation  {}/{}".format(epoch, n_epochs - 1),
                    "loss: {:.4f}".format(val_loss),
                    "acc: {:.4f}".format(val_acc),
                    "f1 score: {:.4f}".format(val_f1),
                    "AUROC: {:.4f}".format(val_auroc),
                    "AUPRC: {:.4f}".format(val_auprc),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                # if writer is not None:
                #     writer.add_scalar("val_loss", val_loss, epoch)
                #     writer.add_scalar("val_acc", val_acc, epoch)


                if val_acc > val_acc_max:
                    print("acc ({:.6f} --> {:.6f})".format(val_acc_max, val_acc))
                    val_acc_max = val_acc
                    b_new_best = True

        if args.rank == 0 and args.logdir is not None:
            save_checkpoint(model, epoch, args, best_acc=val_acc, filename="model_final_over.pt")
            if b_new_best:
                print("Copying to model.pt new best model!!!!")
                shutil.copyfile(os.path.join(args.logdir, "model_final_over.pt"), os.path.join(args.logdir, "model_over.pt"))

        scheduler.step()

    print("ALL DONE")


def parse_args():

    parser = argparse.ArgumentParser(description="Multiple Instance Learning (MIL) example of classification from WSI.")
    parser.add_argument(
        "--data_root", default="/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK", help="path to root folder of images"
    )
    #parser.add_argument("--dataset_json", default='lymph_1014_svs.json', type=str, help="path to dataset json file")
    parser.add_argument("--dataset_json", default='lymph_newdata.json', type=str, help="path to dataset json file")

    parser.add_argument("--num_classes", default=1, type=int, help="number of output classes")
    parser.add_argument("--mil_mode", default="att_trans", help="MIL algorithm")
    parser.add_argument(
        "--max_tile", default=50, type=int, help="to prevent GPU memory error"
    )
    parser.add_argument("--train_patch", default = 20, type = int)
    parser.add_argument("--tile_size", default=256, type=int, help="size of square patch (instance) in pixels")

    parser.add_argument("--checkpoint", default=None, help="load existing checkpoint")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run only inference on the validation set, must specify the checkpoint argument",
    )

    parser.add_argument("--logdir", default=None, help="path to log directory to store Tensorboard logs")

    parser.add_argument("--epochs", default=25, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size, the number of WSI images per gpu")
    parser.add_argument("--optim_lr", default=3e-5, type=float, help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float, help="optimizer weight decay")
    parser.add_argument("--amp", action="store_true", help="use AMP, recommended")
    parser.add_argument(
        "--val_every",
        default=1,
        type=int,
        help="run validation after this number of epochs, default 1 to run every epoch",
    )
    parser.add_argument("--workers", default=4, type=int, help="number of workers for data loading")

    ###for multigpu
    parser.add_argument("--distributed", action="store_true", help="use multigpu training, recommended")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument(
        "--dist-url", default="tcp://localhost:23456", type=str, help="url used to set up distributed training"
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

    parser.add_argument(
        "--quick", action="store_true", help="use a small subset of data for debugging"
    )  # for debugging

    args = parser.parse_args()

    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")

    return args

if __name__ == "__main__":
    random.seed(0)
    args = parse_args()
    args.amp =True
    #args.logdir = 'nfs/thena/shared/checkpoints'
    #args.checkpoint = 'nfs/thena/shared/checkpoints/model_over.pt'
    #########
    args.distributed = False
    #args.quick = False
    #args.validate = True


    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.optim_lr = ngpus_per_node * args.optim_lr / 2  # heuristic to scale up learning rate in multigpu setup
        args.world_size = ngpus_per_node * args.world_size

        print("Multigpu", ngpus_per_node, "rescaled lr", args.optim_lr)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)
