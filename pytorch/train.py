"""
"""
#from builtins import range
import os
import sys
sys.path.append(os.getcwd())
import _init_paths
from PointNetFCAE import *
#from modules.emd import EMDModule
from utils.chamfer.dist_chamfer import chamferDist as chamfer
from tools.obs_data_loader import load_dataset
from tools.import_tool import fileImport
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
#from tools.path_data_loader import load_dataset_end2end
from torch.autograd import Variable
import time

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    importer = fileImport()
    env_data_path = args.env_data_path
    path_data_path = args.path_data_path
    pcd_data_path = args.pointcloud_data_path
    # append all envs and obstacles
    #envs_files = os.listdir(env_data_path)
    #envs_files = ['trainEnvironments.pkl']
    envs_files = ['trainEnvironmentsLarge.pkl']
    #envs_files = ['trainEnvironments.pkl']
    obstacles = []
    for envs_file in envs_files:
        envs = importer.environments_import(env_data_path + envs_file)

        print("Loading obstacle data...\n")
        obs = load_dataset(envs, pcd_data_path, importer)
        obstacles.append(obs)


    obstacles = np.stack(obstacles).astype(float)[0].reshape(len(obs),-1,3)
    print(obstacles.shape)
    print("Loaded dataset, targets, and pontcloud obstacle vectors: ")
    print("\n")

    if not os.path.exists(args.trained_model_path):
        os.makedirs(args.trained_model_path)

    # Build the models
    net = PointNetFCAE(code_ntfs=1024, num_points=len(obstacles[0]), output_channels=3)
    if torch.cuda.is_available():
        net.cuda()


    # Loss and Optimizer
    params = list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    total_loss = []
    epoch = 1

    sm = 100  # start saving models after 100 epochs

    print("Starting epochs...\n")
    # epoch=1
    done = False
    for epoch in range(args.num_epochs):
        # while (not done)
        # every time use a new obstacle
        start = time.time()
        print("epoch" + str(epoch))
        avg_loss = 0
        for i in range(0, len(obstacles), args.batch_size):
            # Forward, Backward and Optimize
            # zero gradients
            net.zero_grad()
            # convert to pytorch tensors and Varialbes
            bobs = torch.from_numpy(obstacles[i:i+args.batch_size]).type(torch.FloatTensor)
            #bobs = to_var(bobs).view(len(bobs), -1, 3).permute(0,2,1)
            bobs = to_var(bobs)
            # forward pass through encoder
            bt = net(bobs)
            # compute overall loss and backprop all the way
            loss1, loss2 = chamfer()(bobs, bt)
            #loss1, loss2 = criterion(bobs, bt)
            print('loss1')
            print(loss1)
            print('loss2')
            print(loss2)
            loss = torch.mean(loss1) + torch.mean(loss2)
            print('loss:')
            print(loss)
            avg_loss = avg_loss+loss.data
            loss.backward()
            optimizer.step()

        print("--average loss:")

        # Save the models
        if epoch == sm:
            print("\nSaving model\n")
            print("time: " + str(time.time() - start))
            torch.save(net.state_dict(), os.path.join(
                args.trained_model_path, 'pointnet_'+str(epoch)+'.pkl'))
            #if (epoch != 1):
            sm = sm+100  # save model after every 50 epochs from 100 epoch ownwards

    torch.save(total_loss, 'total_loss.dat')
    print(encoder.state_dict())
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--path_data_path', type=str, default='./data/train/paths/')
    parser.add_argument('--pointcloud_data_path', type=str, default='./data/train/pcd/')
    parser.add_argument('--trained_model_path', type=str, default='./models/sample_train/', help='path for saving trained models')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=74)
    parser.add_argument('--mlp_output_size', type=int, default=7)

    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--path_data_file', type=str, default='trainPaths.pkl')

    args = parser.parse_args()
    main(args)
