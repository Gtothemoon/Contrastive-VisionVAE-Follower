import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


torch.manual_seed(42)
lr_list = []
model = Net()
lr = 0.1
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
x = torch.randn((1, 5), dtype=torch.float32)
y = torch.tensor([[2, 4, 8]], dtype=torch.float32)

k = 0
for i in range(0, 20000, 100):
    # print('k: {}, i: {}'.format(k, i))
    # k += 1
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
for i, j in enumerate(lr_list):
    print('{}: {}'.format(i, lr_list[i]))
plt.plot(range(200), lr_list, color='r')
plt.show()

# def train():
#     optimizer.zero_grad()
#     output = model(x)
#     loss = criterion(output, y)
#     # print('loss: {}, output: {}'.format(loss, output))
#     loss.backward()
#     optimizer.step()


# for epoch in range(20000):
#     train()
#     for _ in range(2):
#         scheduler.step()  # step_size的次数根据scheduler.step()的调用次数确定
#         lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
#
# for i, j in enumerate(lr_list):
#     print('{}: {}'.format(i + 1, lr_list[i]))
# plt.plot(range(40000), lr_list, color='r')
# plt.show()


# """tqdm进度条的使用"""
# from tqdm import tqdm
# import time
#
# for i in tqdm(range(100)):
#     time.sleep(0.1)
#     pass
