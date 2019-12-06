import nocd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

from sklearn.preprocessing import normalize

# %matplotlib inline
data_path = "./data/"
file_name = "facebook_ego/fb_3437"
file_name = "sina"
# file_name = "mag_med"


torch.set_default_tensor_type(torch.cuda.FloatTensor)

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


# x_norm = normalize(X)  # node features
# x_norm = normalize(A)  # adjacency matrix
x_norm = sp.hstack([normalize(X), normalize(A)])  # 连接A和X，用的是水平拼接。

# x_norm = nocd.utils.to_sparse_tensor(x_norm).cuda()

if os.path.exists("./trained/" + file_name):
    gnn = torch.load("./trained/" + file_name)
    model_saver = nocd.train.ModelSaver(gnn)
    adj_norm = gnn.normalize_adj(A)
else:
    hidden_sizes = [128]  # hidden sizes of the GNN
    weight_decay = 1e-5  # strength of L2 regularization on GNN weights
    lr = 1e-3  # learning rate
    max_epochs = 500  # number of epochs to train
    display_step = 25  # how often to compute validation loss
    balance_loss = True  # whether to use balanced loss
    stochastic_loss = False  # whether to use stochastic or full-batch training
    batch_size = 20000  # batch size (only for stochastic training)

    x_norm = nocd.utils.to_sparse_tensor(x_norm).cuda()

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

    if __name__ == '__main__':
        for epoch, batch in enumerate(sampler):
            if epoch > max_epochs:
                break
            if epoch % 25 == 0:
                with torch.no_grad():
                    gnn.eval()
                    # Compute validation loss
                    Z = F.relu(gnn(x_norm, adj_norm))
                    val_loss = decoder.loss_full(Z, A)
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
                loss = decoder.loss_full(Z, A)
            # loss += nocd.utils.l2_reg_loss(gnn, scale=weight_decay)
            loss.backward()
            optimizer.step()

# torch.save(gnn, "./trained/" + file_name)

# plt.hist(Z[Z > 0].cpu().detach().numpy(), 100)
thresh = 0.1
new_node_x = torch.from_numpy(np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    dtype=np.int64))

new_node_adj = torch.from_numpy(np.array([[349, 349, 349, 349, 349, 349, 349, 349, 349],
                                          [317, 322, 26, 31, 168, 124, 285, 255, 129]], dtype=np.int64))

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
avg_nmi = list()
avg_f1_score = list()
avg_f2_score = list()

for thresh in np.arange(0.1, 1.0, 0.1):
    # model_saver.restore()
    gnn.eval()
    Z = F.relu(gnn(x_norm, adj_norm))
    np.save("z.npy", Z.cpu().detach().numpy())

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
    plt.figure(figsize=[10, 10])
    z = np.argmax(Z_pred, 1)
    o = np.argsort(z)
    nocd.utils.plot_sparse_clustered_adjacency(A, K, z, o, markersize=0.05)

print('------------------------------')
print("average nmi: ", np.average(avg_nmi))
print("average F-1 score: ", np.average(avg_f1_score))
print("average F-2 score: ", np.average(avg_f2_score))
print('------------------------------')
print("maximum nmi: ", np.max(avg_nmi))
print("maximum F-1 score: ", np.max(avg_f1_score))
print("maximum F-2 score: ", np.max(avg_f2_score))

"""
# Sizes of detected communities
print(Z_pred.sum(0))

density_baseline = A.nnz / (N ** 2 - N)
num_triangles = (A @ A @ A).diagonal().sum() / 6
num_possible_triangles = (N - 2) * (N - 1) * N / 6
clust_coef_baseline = num_triangles / num_possible_triangles
print(f'Background (over the entire graph):\n'
      f' - density    = {density_baseline:.3e}\n'
      f' - clust_coef = {clust_coef_baseline:.3e}')

metrics = nocd.metrics.evaluate_unsupervised(Z_gt, A)
print(f"Ground truth communities:\n"
      f" - coverage    = {metrics['coverage']:.4f}\n"
      f" - conductance = {metrics['conductance']:.4f}\n"
      f" - density     = {metrics['density']:.3e}\n"
      f" - clust_coef  = {metrics['clustering_coef']:.3e}")

metrics = nocd.metrics.evaluate_unsupervised(Z_pred, A)
print(f"Predicted communities:\n"
      f" - coverage    = {metrics['coverage']:.4f}\n"
      f" - conductance = {metrics['conductance']:.4f}\n"
      f" - density     = {metrics['density']:.3e}\n"
      f" - clust_coef  = {metrics['clustering_coef']:.3e}")
"""
