import torch as th
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

# device = th.device("cuda" if th.cuda.is_available() else "cpu")
# print(th.cuda.get_device_name(device))

device = th.device("cpu")


class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SimpleFCNN, self).__init__()
        self.relu = nn.ReLU()
        # initialize the first layer
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.num_layers = num_layers
        # initialize the layers
        for i in range(num_layers - 2):
            setattr(self, f"fc{i}", nn.Linear(hidden_size, hidden_size))

        # initialize last layer extra
        self.last_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # apply first layer
        x = self.relu(self.first_layer(x))
        # iterate over all layers but the last one
        for i in range(self.num_layers - 2):
            x = self.relu(getattr(self, f"fc{i}")(x))
        # apply last layer without relu
        x = self.last_layer(x)
        return x


class CouplingLayer(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers):
        super(CouplingLayer, self).__init__()
        # Initialize the fully connected network
        upper_split_size = math.ceil(num_features / 2)
        lower_split_size = num_features - upper_split_size
        # self.t = SimpleFCNN(
        #     upper_split_size, hidden_size, 2*lower_split_size, num_layers
        # )
        self.s = SimpleFCNN(
            upper_split_size, hidden_size, 2 * lower_split_size, num_layers
        )

    def forward(
        self,
        x,
    ):
        x_1, x_2 = th.tensor_split(x, 2, dim=1)

        s, t = th.tensor_split(self.s(x_1), 2, dim=1)
        # Forward pass through the coupling layer
        x_2 = th.abs(s) * x_2 + t
        log_det_J = th.sum(th.log(th.abs(s)), dim=1)
        return th.cat((x_1, x_2), dim=1), log_det_J

    def backward(self, x):
        x_1, x_2 = th.tensor_split(x, 2, dim=1)
        s, t = th.tensor_split(self.s(x_1), 2, dim=1)
        # Forward pass through the coupling layer
        x_2 = (x_2 - t) / th.abs(s)
        return th.cat((x_1, x_2), dim=1)


class Flow(nn.Module):
    def __init__(self, num_blocks, num_features, hidden_size, num_layers):
        super(Flow, self).__init__()

        self.num_blocks = num_blocks

        for i in range(num_blocks):
            setattr(
                self,
                f"cl_{i}",
                CouplingLayer(num_features, hidden_size, num_layers),
            )

    def forward(self, x):
        log_det_J = 0
        for i in range(self.num_blocks):
            x, _log_det_J = getattr(self, f"cl_{i}").forward(x)
            x = x[:, [1, 0]]
            log_det_J += _log_det_J

        return x, log_det_J

    def forward_n_layers(self, x, n):
        for i in range(n):
            x, _log_det_J = getattr(self, f"cl_{i}").forward(x)
            x = x[:, [1, 0]]

        return x

    def backward(self, x):
        for i in range(self.num_blocks - 1, -1, -1):
            x = x[:, [1, 0]]
            x = getattr(self, f"cl_{i}").backward(x)
        return x

    def backward_n_layers(self, x, n):
        for i in range(self.num_blocks - 1, self.num_blocks - 1 - n, -1):
            x = x[:, [1, 0]]
            x = getattr(self, f"cl_{i}").backward(x)
        return x


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def Isotropic_Gaussian(self, x):
        xx = th.sum(x**2, axis=1)
        return -0.5 * (xx - 2 * th.log(th.tensor(2 * th.pi)))

    def forward(self, x, log_det_J, model, lambda_1=1e-3):
        mse_penalty = th.nn.MSELoss(size_average=False)
        reg_loss = 0
        for param in model.parameters():
            reg_loss += mse_penalty(param, th.zeros_like(param))  #
        return -th.sum(
            self.Isotropic_Gaussian(x) + log_det_J
        ) + lambda_1 * reg_loss, -th.sum(self.Isotropic_Gaussian(x) + log_det_J)


x = []
distance = 0.4
sample_multiplier = 5
# 16
# for i in np.arange(0, 8, distance):
#     for j in np.arange(0, 2, distance):
#         x.append([j, i])
for i in range(16 * sample_multiplier):
    x.append([np.random.uniform(0, 2), np.random.uniform(0, 8)])

# 4
# for i in np.arange(0, 2, distance):
#     for j in np.arange(2, 6, distance):
#         x.append([j, i])
for i in range(4 * sample_multiplier):
    x.append([np.random.uniform(2, 4), np.random.uniform(0, 2)])


# 16
# for i in np.arange(0, 8, distance):
#     for j in np.arange(10, 12, distance):
#         x.append([j, i])
for i in range(16 * sample_multiplier):
    x.append([np.random.uniform(10, 12), np.random.uniform(0, 8)])

# 8
# for i in np.arange(6, 8, distance):
#     for j in np.arange(12, 16, distance):
#         x.append([j, i])
for i in range(8 * sample_multiplier):
    x.append([np.random.uniform(12, 16), np.random.uniform(6, 8)])

# 8
# for i in np.arange(3, 5, distance):
#     for j in np.arange(12, 16, distance):
#         x.append([j, i])
for i in range(8 * sample_multiplier):
    x.append([np.random.uniform(12, 16), np.random.uniform(3, 5)])

# 8
# for i in np.arange(0, 2, distance):
#     for j in np.arange(12, 16, distance):
#         x.append([j, i])
for i in range(8 * sample_multiplier):
    x.append([np.random.uniform(12, 16), np.random.uniform(0, 2)])

x = th.tensor(x)
x = x.to(device)

num_features = 2  # Number of features in the input
hidden_size = 32  # Number of hidden units in the FCNN
num_layers = 16  # Number of layers in the FCNN
num_blocks = 6  # Number of coupling layers in the flow

# Initialize the flow
flow = Flow(num_blocks, num_features, hidden_size, num_layers)

# Initialize the loss
lossfunc = Loss()
flow.to(device)
loss_arr = []


optimizer = th.optim.Adam(
    [
        {"params": flow.parameters(), "lr": 1e-3},
    ]
)
dataloader = th.utils.data.DataLoader(x, batch_size=50000, shuffle=True)
sheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.1)
for i in range(5000):
    epochloss = 0
    epochcounter = 0
    epochcleanedloss = 0
    for x_batch in dataloader:
        optimizer.zero_grad()
        loss = 0.0
        y_pred, log_det_J = flow.forward(x_batch.float())
        loss, cleaned_loss = lossfunc.forward(y_pred.float(), log_det_J, flow)
        epochloss += loss
        epochcleanedloss += cleaned_loss
        epochcounter += 1
        loss_arr.append(cleaned_loss.cpu().item())
        loss.backward()

        optimizer.step()
        sheduler.step()
    if i % 1000 == 0:
        print(
            "Epoch: "
            + str(i)
            + " Loss: "
            + str(epochloss.item() / epochcounter)
            + " Cleaned Loss: "
            + str(epochcleanedloss.item() / epochcounter)
        )

plt.figure()
plt.plot(loss_arr)
plt.ylim(0, 450)
plt.xlabel("Epoch")
plt.ylabel("Negative Log Likelihood")
plt.savefig("Experiment1_loss.png")

samples = th.tensor(np.random.normal([0, 0], 1, size=(1000, 2))).to(device)
# plt.scatter(samples[:, 0], samples[:, 1], label='Original Data')
result = flow.backward(samples.float()).cpu().detach().numpy()
# result = np.where(result < 10, result, -1)

samples = samples.cpu()

import matplotlib.pyplot as plt

col_red = "#c61826"
col_dark_red = "#590d08"
col_blue = "#01024d"

plt.figure()
plt.scatter(result[:, 0], result[:, 1], s=25, label="Transformed Data", color=col_red)
plt.legend(fontsize=8)
plt.gca().set(xlim=(-2, 18), ylim=(-2, 10), xlabel="X", ylabel="Y")
plt.savefig("Experiment1_trans_data.png")
x = x.cpu()
plt.figure()
plt.scatter(x[:, 0], x[:, 1], s=25, label="Original Data", alpha=1, color=col_blue)
plt.legend(fontsize=8)
plt.gca().set(xlim=(-2, 18), ylim=(-2, 10), xlabel="X", ylabel="Y")
plt.savefig("Experiment1_orig_data.png")

plt.figure()
plt.scatter(x[:, 0], x[:, 1], s=25, label="Original Data", alpha=1, color=col_blue)
plt.scatter(
    result[:, 0], result[:, 1], s=25, label="Transformed Data", alpha=1, color=col_red
)
plt.legend(fontsize=8)
plt.gca().set(xlim=(-2, 18), ylim=(-2, 10), xlabel="X", ylabel="Y")
plt.savefig("Experiment1_both_data.png")

samples = th.tensor(np.random.normal([0, 0], 1, size=(1000, 2))).to(device)
switch = False
for i in range(num_blocks + 1):
    plt.figure()
    result = flow.backward_n_layers(samples.float(), i).cpu().detach().numpy()
    samples = samples.cpu()
    plt.scatter(
        samples[:, 0],
        samples[:, 1],
        s=25,
        label="Original Distribution",
        alpha=1,
        color=col_blue,
    )
    if switch:
        plt.scatter(
            result[:, 1],
            result[:, 0],
            s=25,
            label="Transformed Distribution",
            color=col_red,
        )
    else:
        plt.scatter(
            result[:, 0],
            result[:, 1],
            s=25,
            label="Transformed Distribution",
            color=col_red,
        )

    plt.legend(fontsize=8)
    plt.gca().set(xlim=(-2, 18), ylim=(-2, 10), xlabel="X", ylabel="Y")
    plt.savefig(f"Experiment1_block{i}_data.png")
    samples = samples.to(device)
    switch = not switch


def MMD(x, y):
    gamma = 2
    xx, yy, zz = th.mm(x, x.t()), th.mm(y, y.t()), th.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    rxx = rx[0].repeat(y.shape[0], 1)
    ryy = ry[0].repeat(x.shape[0], 1)
    dxy = rxx.t() + ryy - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        th.zeros(xx.shape),
        th.zeros(yy.shape),
        th.zeros(zz.shape),
    )
    XX += 1 / (1 + dxx / gamma**2)
    YY += 1 / (1 + dyy / gamma**2)
    XY += 1 / (1 + dxy / gamma**2)
    return XX.mean() + YY.mean() - 2 * XY.mean()


print(
    "MMD after transformation:",
    MMD(x.cpu(), th.tensor(result).cpu().float()).detach().numpy(),
)
print(
    "MMD before transformation:",
    MMD(x.cpu(), th.tensor(samples).cpu().float()).detach().numpy(),
)
