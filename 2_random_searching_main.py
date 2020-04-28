import nocd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from nocd.recom_metrics import *
from sklearn.preprocessing import normalize
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from sklearn.gaussian_process.kernels import Matern
import time
import random

import matplotlib.pyplot as plt

data_path = "./data/"
file_name = "facebook_ego/fb_698"
file_name = "sina"
torch.set_default_tensor_type(torch.cuda.FloatTensor)

start_time = time.time()


# plotting function
def plot_rs(lower_bound, upper_bound, train_features, train_labels):
    plt.clf()
    plt.cla()
    plt.close()
    plt.style.use(['classic'])
    plt.figure(figsize=(16, 10))
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.suptitle('Gaussian Process and Utility Function After {} Steps'.format(len(train_features)),
    #             fontdict={'size': 20})
    # 主图的绘制过程
    plt.plot(train_features.flatten(), train_labels, 'D', markersize=8, label=u'Observations', color='r')

    # 给散点图标点
    for i in range(len(train_features)):
        plt.annotate("(" + str(round(float(train_features[i]), 2)) + ", " + str(round(train_labels[i], 2)) + ")",
                     (train_features[i], train_labels[i]))

    distance = np.max(train_labels) - np.min(train_labels)
    plt.axis([lower_bound - 0.1,
              upper_bound + 0.1,
              np.min(train_labels) - 0.2 * distance,
              np.max(train_labels) + 0.2 * distance])
    plt.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    return random.uniform(lower_bound, upper_bound)


if file_name == "sina":
    loader = np.load(data_path + file_name + '.npz', allow_pickle=True)
    A, X, Z_gt = loader['A'], loader['X'], loader['Z']
    A = sp.csr_matrix(A)
    X = sp.csr_matrix(X)
else:
    loader = nocd.data.load_dataset(data_path + file_name + '.npz')
    A, X, Z_gt = loader['A'], loader['X'], loader['Z']

# 这里，A是adjacency matrix， X是node features， Z是community的ground truth

N, K = Z_gt.shape
# 这里，N是Node的数量，K是community的数量


# 自动化学习的相关参数
lower_bound = 0.4
upper_bound = 3
input_linespace = np.linspace(lower_bound, upper_bound, 10000).reshape(-1, 1)
train_features = list()
train_labels = list()
round_counter = 0

# 最后的总结性输出
average_NMI_list = list()
average_F1_list = list()
average_F2_list = list()
maximum_NMI_list = list()
maximum_F1_list = list()
maximum_F2_list = list()
local_MAE_list = list()
local_RMSE_list = list()
global_MAE_list = list()
global_RMSE_list = list()

alpha = random.uniform(lower_bound, upper_bound)  # 我们将初始的hyper parameter值设置为1，方便训练
num_of_iter = 10

while True:
    round_counter += 1
    A, X, Z_gt = loader['A'], loader['X'], loader['Z']
    if file_name == "sina":
        A = sp.csr_matrix(A)
        X = sp.csr_matrix(X)
    hidden_sizes = [128]  # hidden sizes of the GNN
    weight_decay = 1e-5  # strength of L2 regularization on GNN weights
    lr = 1e-3  # learning rate
    max_epochs = 500  # number of epochs to train
    display_step = 25  # how often to compute validation loss
    balance_loss = True  # whether to use balanced loss
    stochastic_loss = False  # whether to use stochastic or full-batch training
    batch_size = 20000  # batch size (only for stochastic training)

    x_norm = sp.hstack([normalize(X), normalize(A)])  # 连接A和X，用的是水平拼接。
    x_norm = nocd.utils.to_sparse_tensor(x_norm).cuda()
    # x_norm = nocd.utils.to_sparse_tensor(x_norm).cuda()

    sampler = nocd.sampler.get_edge_sampler(A, batch_size, batch_size, num_workers=0)
    # sampler here is a data loader.
    # If using old model, you should also set weight_decay=1e-2
    # gnn = nocd.nn.GCN(x_norm.shape[1], hidden_sizes, K, batch_norm=True).cuda()

    # 这里的GNN是一个三层的网络，输入是层是x_norm的宽度，输出是K个，即与ground truth相同
    gnn = nocd.nn.ImprovedGCN(x_norm.shape[1], hidden_sizes, K).cuda()

    """Normalize adjacency matrix and convert it to a sparse tensor."""
    adj_norm = gnn.normalize_adj(A)

    decoder = nocd.nn.BerpoDecoder(N, A.nnz, balance_loss=balance_loss)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

    # 设置初始的loss为infinity
    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = nocd.train.NoImprovementStopping(validation_fn, patience=10)
    model_saver = nocd.train.ModelSaver(gnn)

    for epoch, batch in enumerate(sampler):
        if epoch > max_epochs:
            break
        if epoch % 25 == 0:
            with torch.no_grad():
                gnn.eval()
                # Compute validation loss
                Z = F.relu(gnn(x_norm, adj_norm))
                val_loss = decoder.loss_full(Z, A, alpha)
                print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}')

                # Check if it's time for early stopping / to save the model
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
                if early_stopping.should_stop():
                    print(f'Breaking due to early stopping at epoch {epoch}')
                    break

        # Training step
        gnn.train()

        # zero the parameter gradients
        optimizer.zero_grad()

        Z = F.relu(gnn(x_norm, adj_norm))

        ones_idx, zeros_idx = batch
        if stochastic_loss:
            # 随机loss
            loss = decoder.loss_batch(Z, ones_idx, zeros_idx)
        else:
            # 非随机loss
            loss = decoder.loss_full(Z, A, alpha)
        # loss += nocd.utils.l2_reg_loss(gnn, scale=weight_decay)
        loss.backward()
        optimizer.step()

    # torch.save(gnn, "./trained/" + file_name)

    # plt.hist(Z[Z > 0].cpu().detach().numpy(), 100)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_nmi = list()
    avg_f1_score = list()
    avg_f2_score = list()

    for thresh in np.arange(0.1, 1.0, 0.1):
        # model_saver.restore()
        gnn.eval()
        Z = F.relu(gnn(x_norm, adj_norm))
        np.save("z.npy", Z.cpu().detach().numpy())
        np.save("A.npy", A.A)

        # print(Z.cpu().detach().numpy().shape, Z.cpu().detach().numpy()[0])
        Z_pred = Z.cpu().detach().numpy() > thresh

        nmi = nocd.metrics.overlapping_nmi(Z_pred, Z_gt)
        f1_score = nocd.metrics.f_evaluate(nocd.utils.Z_translator(Z_pred), nocd.utils.Z_translator(Z_gt), 1)
        f2_score = nocd.metrics.f_evaluate(nocd.utils.Z_translator(Z_pred), nocd.utils.Z_translator(Z_gt), 2)
        avg_nmi.append(nmi)
        avg_f1_score.append(f1_score)
        avg_f2_score.append(f2_score)
        print('------------------------------')
        print("Current threshold: " + str(round(thresh, 3)))
        print(f'nmi = {nmi:.5f}')
        print(f'F-1 Score = {f1_score:.5f}')
        print(f'F-2 Score = {f2_score:.5f}')
        # print(Z_pred.sum(0))
        # print(Z_gt.sum(0))
        # plt.figure(figsize=[10, 10])
        z = np.argmax(Z_pred, 1)
        o = np.argsort(z)
        nocd.utils.plot_sparse_clustered_adjacency(A, K, z, o, markersize=0.05)

    # 所有的挖掘指标
    average_NMI_list.append(np.average(avg_nmi))
    average_F1_list.append(np.average(avg_f1_score))
    average_F2_list.append(np.average(avg_f2_score))

    maximum_NMI_list.append(np.max(avg_nmi))
    maximum_F1_list.append(np.max(avg_f1_score))
    maximum_F2_list.append(np.max(avg_f2_score))

    # 所有的推荐指标
    emb = Z.cpu().detach().numpy()
    ori_adj = A.A

    adj_local = comm_to_adj(emb, ori_adj.shape[0], A.data.nonzero()[0].shape[0])
    local_MAE_list.append(MAE(adj_local, ori_adj))
    local_RMSE_list.append(RMSE(adj_local, ori_adj))

    # adj_global = comm_to_adj_globally(emb, ori_adj.shape[0], A.data.nonzero()[0].shape[0])
    # global_MAE_list.append(MAE(adj_global, ori_adj))
    # global_RMSE_list.append(RMSE(adj_global, ori_adj))

    print('------------------------------')
    print("average nmi: ", average_NMI_list[-1])
    print("average F-1 score: ", average_F1_list[-1])
    print("average F-2 score: ", average_F2_list[-1])
    print('------------------------------')
    print("maximum nmi: ", maximum_NMI_list[-1])
    print("maximum F-1 score: ", maximum_F1_list[-1])
    print("maximum F-2 score: ", maximum_F2_list[-1])

    print('------------------------------')
    print("MAE:", local_MAE_list[-1])
    print("RMSE:", local_RMSE_list[-1], "\n")
    # print("MAE:", global_MAE_list[-1])
    # print("RMSE:", global_RMSE_list[-1])

    # 所有的自动化训练模块
    train_features = list(train_features)
    train_features.append(alpha)
    train_features = np.array(train_features).reshape(-1, 1)
    train_labels = list(train_labels)
    train_labels.append(maximum_NMI_list[-1])
    train_labels = np.array(train_labels)

    # 重新设置下一轮的alpha
    alpha = plot_rs(lower_bound, upper_bound, train_features, train_labels)
    print("Next Guess: ", alpha)
    plt.show()
    if round_counter >= num_of_iter:
        break

index = int(np.argmax(maximum_NMI_list))
print('------------------------------')
print('---------Final Results--------')
print('------------------------------')
print("Final average nmi: ", average_NMI_list[index])
print("Final average F-1 score: ", average_F1_list[index])
print("Final average F-2 score: ", average_F2_list[index])
print('------------------------------')
print("Final maximum nmi: ", maximum_NMI_list[index])
print("Final maximum F-1 score: ", maximum_F1_list[index])
print("Final maximum F-2 score: ", maximum_F2_list[index])

print('------------------------------')
print("Final MAE:", local_MAE_list[index])
print("Final RMSE:", local_RMSE_list[index], "\n")
# print("Final MAE:", global_MAE_list[index])
# print("Final RMSE:", global_RMSE_list[index])

print('------------------------------')
print('-----------Summary------------')
print('------------------------------')
end_time = time.time()
print('Time cost in total: ', end_time - start_time, 's')
print('Iteration rounds: ', str(len(train_features)))
print("Parameter selected: ", train_features[index])
