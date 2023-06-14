import os
import shutil
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model import Decoder
from utils import SdfDataset, mkdir_p, isdir, showMeshReconstruction, create_training_dataset, split_train_val, L1_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint_{}.pth.tar'.format(state['epoch']))
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def train(dataset, model, optimizer, args):

    model.train()

    Δ = args.clamping_distance
    loss_sum = 0.0
    loss_count = 0.0
    num_batch = len(dataset)

    for i in range(num_batch):
        batch = dataset[i]
        s = batch['gt_sdf'].to(device)
        f_p = model(batch['xyz'].to(device))
        loss = torch.sum(L1_loss(f_p, s, Δ)) # Note L1_loss defined in utils.py:71
        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        loss_count += batch['xyz'].shape[0]

    return loss_sum.item() / loss_count


def val(dataset, model, optimizer, args):

    model.eval()

    Δ = args.clamping_distance
    loss_sum = 0.0
    loss_count = 0.0
    num_batch = len(dataset)

    for i in range(num_batch):
        batch = dataset[i]
        with torch.no_grad():
            f_p = model(batch['xyz'].to(device))
            s = batch['gt_sdf'].to(device)
            loss_sum += torch.sum(L1_loss(f_p, s, Δ)) # Note L1_loss defined in utils.py:71
            loss_count += batch['xyz'].shape[0]

    return loss_sum.item() / loss_count



def test(dataset, model, args):

    model.eval()

    num_batch = len(dataset)
    number_samples = dataset.number_samples
    grid_shape = dataset.grid_shape
    IF = np.zeros((number_samples, ))
    start_idx = 0
    for i in range(num_batch):
        data = dataset[i]  # a dict
        xyz_tensor = data['xyz'].to(device)
        this_bs = xyz_tensor.shape[0]
        end_idx = start_idx + this_bs
        with torch.no_grad():
            pred_sdf_tensor = model(xyz_tensor)
            pred_sdf_tensor = torch.clamp(pred_sdf_tensor, -args.clamping_distance, args.clamping_distance)
        pred_sdf = pred_sdf_tensor.cpu().squeeze().numpy()
        IF[start_idx:end_idx] = pred_sdf
        start_idx = end_idx

    IF = np.reshape(IF, grid_shape)

    verts, triangles = showMeshReconstruction(IF)

    with open('test.obj', 'w') as outfile:
        for v in verts:
            outfile.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in triangles:
            outfile.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")            

    return

def main(args):

    best_loss = 2e10
    best_epoch = -1

    if not isdir(args.checkpoint_folder):
        print("Creating new checkpoint folder " + args.checkpoint_folder)
        mkdir_p(args.checkpoint_folder)

    model = Decoder(args).to(device)

    print("=> Using " + device.type + " device.")

    cudnn.benchmark = True

    if args.evaluate:
        path_to_resume_file = os.path.join(args.checkpoint_folder, args.resume_file)
        print(f"\nEvaluation only\n=> Loading training checkpoint '{path_to_resume_file}'")
        checkpoint = torch.load(path_to_resume_file)
        model.load_state_dict(checkpoint['state_dict'])
        test_dataset = SdfDataset(phase='test', args=args)
        test(test_dataset, model, args)
        return

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    print(f"=> Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    dataset = create_training_dataset(file=args.input_pts)
    print(f"=> Number of points in input point cloud: {dataset['points'].shape[0]}")
    
    split = split_train_val(dataset, args.train_split_ratio)
    train_dataset = SdfDataset(points=split['train_points'], normals=split['train_normals'], args=args)
    val_dataset = SdfDataset(points=split['val_points'], normals=split['val_normals'], phase='val', args=args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)

    for epoch in range(args.start_epoch, args.epochs):

        train_loss = train(train_dataset, model, optimizer, args)
        val_loss = val(val_dataset, model, optimizer, args)

        scheduler.step()

        is_best = val_loss < best_loss

        if is_best:
            best_loss = val_loss
            best_epoch = epoch

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict()
            },
            is_best,
            checkpoint_folder=args.checkpoint_folder
        )
        print(f"Epoch {epoch+1:d}. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}. Best Epoch: {best_epoch+1:d}. Best val loss: {best_loss:.8f}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSDF')

    parser.add_argument("-e", "--evaluate", action="store_true", help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust
    parser.add_argument("--input_pts", default="data/bunny-1000.pts", type=str, help="Input point cloud")
    parser.add_argument("--checkpoint_folder", default="checkpoints/", type=str, help="Folder to save checkpoints")
    parser.add_argument("--resume_file", default="model_best.pth.tar", type=str, help="Path to retrieve latest checkpoint file relative to checkpoint folder")

    # hyperameters of network/options for training
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay/L2 regularization on weights")
    parser.add_argument("--lr", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--schedule", type=int, nargs="+", default=[40, 50], help="Decrease learning rate at these milestone epochs.")
    parser.add_argument("--gamma", default=0.1, type=float, help="Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestone epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="Start from specified epoch number")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--train_batch", default=512, type=int, help="Batch size for training")
    parser.add_argument("--train_split_ratio", default=0.8, type=float, help="ratio of training split")
    parser.add_argument("--N_samples", default=100.0, type=float, help="for each input point, N samples are used for training or validation")
    parser.add_argument("--sample_std", default=0.05, type=float, help="we perturb each surface point along normal direction with mean-zero Gaussian noise with the given standard deviation")
    parser.add_argument("--clamping_distance", default=0.1, type=float, help="clamping distance for sdf")

    # various options for testing and evaluation
    parser.add_argument("--test_batch", default=2048, type=int, help="Batch size for testing")
    parser.add_argument("--grid_N", default=128, type=int, help="construct a 3D NxNxN grid containing the point cloud")
    parser.add_argument("--max_xyz", default=1.0, type=float, help="largest xyz coordinates")

    # custom memory usage argument
    parser.add_argument("-m", "--track_memory", action="store_true", help="Use this flag to get a feel for the memory usage of your model")

    print(parser.parse_args())
    main(parser.parse_args())