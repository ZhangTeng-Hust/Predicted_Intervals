# @Time    : 2022/11/30  21:18
# @Auther  : Teng Zhang
# @File    : ICML_TorchVersion.py
# @Project : Uncertainty
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import os

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 解决中文乱码问题
plt.rcParams["font.size"] = 12  # 设置字体大小

def qd_objective(y_true, y_pred):
    '''Loss_QD-soft, from algorithm 1'''
    y_true = y_true[:, 0]
    y_u = y_pred[:, 0]
    y_l = y_pred[:, 1]
    zero_tensor = torch.tensor(0)
    K_HU = torch.max(zero_tensor, torch.sigmoid(y_u - y_true))
    K_HL = torch.max(zero_tensor, torch.sigmoid(y_true - y_l))
    K_H = torch.mul(K_HU, K_HL)

    K_SU = torch.sigmoid(soften_ * (y_u - y_true))
    K_SL = torch.sigmoid(soften_ * (y_true - y_l))
    K_S = torch.mul(K_SU, K_SL)

    MPIW_c = torch.sum(torch.mul((y_u - y_l), K_H)) / (torch.sum(K_H)+0.001)
    PICP_H = torch.mean(K_H)
    PICP_S = torch.mean(K_S)

    Loss_S = MPIW_c + lambda_ * Batch_size / (alpha_ * (1 - alpha_)) * torch.square(
        torch.max(zero_tensor, (1 - alpha_) - PICP_S))

    return Loss_S


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.FC1 = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU()
        )

        self.predict = nn.Linear(100, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.FC1(x)
        result = self.predict(x)
        return result




if __name__ == '__main__':
    n_samples = 100
    X = np.random.uniform(low=-2., high=2., size=(n_samples, 1))
    y = 1.5 * np.sin(np.pi * X[:, 0]) + np.random.normal(loc=0., scale=1. * np.power(X[:, 0], 2))
    y = y.reshape([-1, 1]) / 5.
    X_train = X
    y_train1 = y
    y_train = np.hstack((y_train1, y_train1))  # make this 2d so will be accepted


    x_grid = np.linspace(-2, 2, 100).reshape(100,1)# for evaluation plots

    # 数据tensor化
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    x_grid = torch.Tensor(x_grid)

    # hyperparameters
    lambda_ = 0.01  # lambda in loss fn
    alpha_ = 0.05  # capturing (1-alpha)% of samples
    soften_ = 160.
    Batch_size = 64  # batch size
    learning_rate = 2e-2
    regularization = 1e-2
    num_epochs = 1000
    result_loss = []

    Model = model()

    print(Model.parameters())
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = torch.nn.MSELoss()
    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=Batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            prediect = Model.forward(batch_x)
            Loss = qd_objective(batch_y, prediect)
            result_loss.append(Loss.data / len(batch_x))
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(epoch, Loss.data)
                y_pred = Model.forward(x_grid)
                y_u_pred = y_pred[:, 0]
                y_l_pred = y_pred[:, 1]
                plt.scatter(X_train.data.numpy(), y_train[:, 0].data.numpy())
                plt.plot(x_grid.data.numpy(), y_u_pred.data.numpy(), color='r',
                         label='Upper')  # upper boundary prediction
                plt.plot(x_grid.data.numpy(), y_l_pred.data.numpy(), color='g',
                         label='Lower')  # lower boundary prediction
                plt.legend(loc = 'upper right')
                plt.ylim(-2, 2)
                plt.title('After training ' + str(epoch) + ' epochs')
                figure_save_path = "figure"
                if not os.path.exists(figure_save_path):
                    os.makedirs(figure_save_path)
                plt.savefig(os.path.join(figure_save_path, str(epoch)))  # 分别命名图片
                plt.close()



    result_loss = np.array(result_loss).reshape(-1)
    x = range(result_loss.shape[0])
    plt.plot(x, result_loss, label='train')
    plt.title('Loss')
    plt.legend()
    plt.show()

    y_pred = Model.forward(X_train)
    y_u_pred = y_pred[:, 0].data.numpy()
    y_l_pred = y_pred[:, 1].data.numpy()
    K_u = y_u_pred > y_train[:, 0].data.numpy()
    K_l = y_l_pred < y_train[:, 0].data.numpy()
    print('PICP:', np.mean(K_u * K_l))
    print('MPIW:', np.round(np.mean(y_u_pred - y_l_pred), 3))
