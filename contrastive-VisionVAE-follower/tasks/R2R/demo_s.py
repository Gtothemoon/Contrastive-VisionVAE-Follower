import argparse
import json
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import warnings
from torch import optim

warnings.filterwarnings('ignore')

angle_inc = np.pi / 6.  # 30°

torch.manual_seed(42)


# def build_viewpoint_loc_embedding(viewIndex):
#     """
#     Position embedding:
#     heading 64D + elevation 64D
#     1) heading: [sin(heading) for _ in range(1, 33)] +
#                 [cos(heading) for _ in range(1, 33)]
#     2) elevation: [sin(elevation) for _ in range(1, 33)] +
#                   [cos(elevation) for _ in range(1, 33)]
#     """
#     embedding = np.zeros((36, 128), np.float32)
#     for absViewIndex in range(36):
#         relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
#         rel_heading = (relViewIndex % 12) * angle_inc
#         rel_elevation = (relViewIndex // 12 - 1) * angle_inc
#         embedding[absViewIndex, 0:32] = np.sin(rel_heading)
#         embedding[absViewIndex, 32:64] = np.cos(rel_heading)
#         embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
#         embedding[absViewIndex, 96:] = np.cos(rel_elevation)
#     return embedding
#
# # pre-compute all the 36 possible paranoram location embeddings
# _static_loc_embeddings = [
#     build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]
# _static_loc_embeddings = np.array(_static_loc_embeddings)
# print(_static_loc_embeddings[0][0])


def load_datasets(splits):
    data = []
    for split in splits:
        with open('data/R2R_sub_%s.json' % split) as f:
            data += json.load(f)
    # {"distance": 12.59, "scan": "VFuaQ6m2Qom", "path_id": 1981,
    # "path": ["5d50911d1c074a3c8de82d35bc4c558b", "8ba5f3cf31934fc98cde3cf4a1116551", "6c8329a5bbd5423696d5d9a22237d0f8", "1ed9136647664140918246b69b5d2dc5", "9377f3ca210946ff9dbea4937cf7d3ad", "8c31225bb638494082b206e492422ebf", "2185b2e2cb704157aefd1dd81f5f3811"],
    # "heading": 3.803,
    # "instructions": ["Walk straight through to doorway on the other side of the room. Turn left and stop on the stairs near the table. ", "Walk straight past the hot tubs.  Go through the doorway and turn to the right.  Then veer to the left and stop there.  Wait. ", "walk forward with the pool on your left. Enter the house and take a right. Go forward into the living room and stop in the living room doorway. "]}
    return data


# data = load_datasets(['train'])
# print(len(data))
# print(data[0]['path'])

class R2RBatch():
    """ Implements the Room to Room navigation task, using discretized viewpoints and pretrained features """

    def __init__(self, batch_size=100, seed=10, splits=['train'], tokenizer=None, beam_size=1,
                 instruction_limit=None):
        # "distance"，"scan"，"path_id"，"path"，"heading"，"instructions"，"instr_id"，"instr_encoding"，"instr_length"
        self.data = []
        self.scans = []
        self.gt = {}  # ground_truth
        load_data = load_datasets(splits)
        for i, item in enumerate(load_data):
            # Split multiple instructions into separate entries
            assert item['path_id'] not in self.gt
            self.gt[item['path_id']] = item
            instructions = item['instructions']
            if instruction_limit:
                instructions = instructions[:instruction_limit]
            for j, instr in enumerate(instructions):  # TODO 把每个样本的3个句子分开
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)  # 例如，"path_id": 1981
                new_item['instructions'] = instr
                new_item['negative'] = load_data[(i + 1) % len(load_data)]  # TODO 添加1个负例
                if tokenizer:  # TODO 训练时有给
                    self.tokenizer = tokenizer
                    new_item['instr_encoding'], new_item['instr_length'] = tokenizer.encode_sentence(instr)
                else:
                    self.tokenizer = None
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)  # 打乱数据
        self.ix = 0
        self.batch_size = batch_size
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))


# batch = R2RBatch()
# print(len(batch.data))
# print(batch.data[0])

# x = np.arange(0, 10, 1)
# print(x)
# a = len(x)
# y = [math.sqrt((a + 0.5 * i) / a) for i in x]
# plt.plot(x, y, c='r')
# plt.show()

# contrast_feat = []
# x = torch.tensor([[1, 2, 3, 4],
#                   [5, 6, 7, 8]])
# y = torch.tensor([[4, 3, 2, 1],
#                   [8, 7, 6, 5]])
# contrast_feat.append(x)
# contrast_feat.append(y)
# # print(contrast_feat)
# contrast_feat = torch.cat(tuple(contrast_feat), 0).reshape()
# print(contrast_feat)
# print(contrast_feat.size())

# """LSTMCell输出的性质"""
# contrast_feat = []  # list方法
# matrix = torch.zeros((10, 4, 3))  # 张量方法
# lstm = nn.LSTMCell(5, 3)
# x = torch.randn((10, 5))
# y = torch.randn((10, 5))
# h_0 = torch.zeros((10, 3))
# c_0 = torch.zeros((10, 3))
# print('x.size():', x.size())
# print('h0.size():', h_0.size())
# print('c0.size():', c_0.size())
# h_1, c_1 = lstm(x, (h_0, c_0))
# h_2, c_2 = lstm(y, (h_1, c_1))
# print('h_1.size(): ', h_1.size())
# print('h_1: ', h_1)
# print('h_2: ', h_2)
# contrast_feat.append(h_1)
# contrast_feat.append(h_2)
# contrast_feat = torch.cat(tuple(contrast_feat), 1).reshape(10, -1, 3)
# matrix[:, 0, :] = h_1
# matrix[:, 1, :] = h_2
# episode_len = 4
# matrix[:, 2, :] = matrix[:, 1, :]
# matrix[:, 3, :] = matrix[:, 1, :]
# print('contrast_feat: ', contrast_feat)
# print('contrast_feat.size(): ', contrast_feat.size())
# print('matrix: ', matrix)
# print('matrix.size(): ', matrix.size())
# print(contrast_feat[0][1])
# print(matrix[0][1])

# """三维张量的转置操作"""
# x = torch.randn(100, 10, 512)
# x = x.transpose(-2, -1)
# print(x.size())

# """交叉熵loss的数据类型"""
# criterion = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# loss = criterion(input, target)
# print('loss: ', loss)
# print('loss.size(): ', loss.size())
# """三维张量(batch x h x w)后两维点积的操作"""
# A = torch.range(1, 5 * 4 * 3, requires_grad=True)
# # print(A)
# A = A.reshape(5, 4, 3)
# # print(A)
# # print('A.size(): ', A.size())
# weight_A = A
# # C = torch.matmul(A, B)
# C = torch.einsum('ijk,ijk->ij', [A, weight_A])
# # print(C)
# # print(C.size())
# weights_C = C
# # contrast_score = torch.einsum('ij,ij->i', [C, weights_C])
# contrast_scores = torch.einsum('ij,ij->i', [C, weights_C]).sum()
# print('contrast_scores: ', contrast_scores)
# print('contrast_scores.size(): ', contrast_scores.size())
#
# sum_loss = loss + contrast_scores
# print('sum_loss: ', sum_loss)
# print('sum_loss.size(): ', sum_loss.size())
# # print('sum_loss.data[0]: ', sum_loss.data[0])
# print('sum_loss.item(): ', sum_loss.item())

# """张量并行计算前的操作"""
# N = 6
# S = 4
# C = 3
# K = 10
# P = 5
# contrast_feat = torch.arange(N * C * P)
# contrast_feat = contrast_feat.reshape(N, C, P)
# # print(x)
# print('contrast_feat[0]: ', contrast_feat[0])
# print('(N x C x P): ', contrast_feat.size())
# contrast_feat = contrast_feat.unsqueeze(0)
# print('(1 x N x C x P): ', contrast_feat.size())
# contrast_feat = contrast_feat.repeat([N, 1, 1, 1])
# print('(N x N x C x P): ', contrast_feat.size())
# print('contrast_feat[0] == x[1] = True')
# print('contrast_feat[0]:', contrast_feat[0])
# print('contrast_feat[1]:', contrast_feat[1])
# print('contrast_feat[1][0]: ', contrast_feat[1][0])
# print('-' * 50)
# ctx_norm = torch.arange(N * S * C)
# ctx_norm = ctx_norm.reshape(N, S, C)
# print('(N x S x C):', ctx_norm.size())
# ctx_norm = ctx_norm.unsqueeze(1)
# print('(N x 1 x S x C):', ctx_norm.size())
# ctx_norm = ctx_norm.repeat([1, K, 1, 1])
# print('(N x K x S x C):', ctx_norm.size())
# print('ctx_norm[0][0] == ctx_norm[0][1] = True')
# print('ctx_norm[0]: ', ctx_norm[0])
# print('ctx_norm[0][0]: ', ctx_norm[0][0])
# print('ctx_norm[0][1]: ', ctx_norm[0][1])

# """nn.functional.normalize()的使用"""
# batch_size = 5
# K = 10
# hidden_size = 4
# episode_len = 3
# queue = torch.arange(batch_size * K * hidden_size * episode_len, requires_grad=False, dtype=torch.float)
# queue = queue.reshape(batch_size, K, hidden_size, episode_len)
# print(queue[0][0])
# queue = nn.functional.normalize(queue, dim=2)
# print(queue[0][0])
# """验证nn.functional.normalize()是否是沿着hidden_size进行的"""
# x = [i for i in range(0, 10, 3)]
# y = [i ** 2 for i in range(0, 10, 3)]
# y = math.sqrt(sum(y))
# z = [i / y for i in x]
# print(z)

# """对比学习scores最后一步的验证"""
# batch_size = 3
# seq_len = 2
# C = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
# print('C: ', C)
# pos_scores = torch.einsum('bs,bs->b', [C, C]).unsqueeze(-1)
# print('pos_scores: ', pos_scores)
# print('pos_scores.size():', pos_scores.size())

# """模型最后取最大预测值及其对应的索引值的方法"""
# x = torch.tensor([4, 5, 7, 2, 1, 3, 2], dtype=torch.float).unsqueeze(0)
# softmax = nn.Softmax(dim=1)
# x = softmax(x)
# print('x:', x)
# print('x.size():', x.size())
# idx = torch.argmax(x, dim=1)
# print('max idx:', idx.item())
# # prob, pred = torch.topk(x, 1, dim=1)
# prob, pred = x.topk(1, dim=1)
# print('prob:', prob)
# print('pred:', pred)

# """CrossEntropyLoss的使用"""
# input = torch.tensor([4, 5, 7, 2, 1, 3, 2], dtype=torch.float).unsqueeze(0)
# criterion = nn.CrossEntropyLoss()
# sm = nn.Softmax(dim=1)
# target = torch.tensor([2])
# loss = criterion(input, target)
# print('loss:', loss)
# input = sm(input)
# sm_loss = criterion(input, target)
# print('loss(softmax):', sm_loss)
#
# """多次做softmax虽然总和仍然为1，但是数值分配会发生改变"""
# x = 0.7
# y = 0.3
# z = math.exp(x) + math.exp(y)
# x = math.exp(x) / z
# y = math.exp(y) / z
# print('x:', x)
# print('y:', y)
# z = math.exp(x) + math.exp(y)
# x = math.exp(x) / z
# y = math.exp(y) / z
# print('x:', x)
# print('y:', y)

# """softmax之后再加权求和的数值范围"""
# x = torch.tensor([100, 20, 10, 30, 90, 120, 40], dtype=torch.float).unsqueeze(0)
# sm = nn.Softmax(dim=-1)
# weight_x = sm(x)
# A = torch.einsum('ij,ij->i', [weight_x, x]).unsqueeze(-1)
# print('x.size():', x.size())
# print('weight_x:', weight_x)
# print(A)
# print(A.size())

# """验证负例的scores计算是否符合逻辑"""
# K = 2
# batch = 2
# seq_len = 2
# episode_len = 2
# hidden_size = 2
# queue = torch.arange(K * hidden_size * episode_len).reshape(K, hidden_size, episode_len)  # (K x hidden_size x e_len)
# print('queue:', queue)
# print('queue.size():', queue.size())
# queue = queue.unsqueeze(0)  # (1 x K x hidden_size x episode_len)
# print('queue:', queue)
# print('queue.size():', queue.size())
# queue = queue.repeat([batch, 1, 1, 1])  # (batch<复制> x K x hidden_size x episode_len)
# print('queue:', queue)
# print('queue.size():', queue.size())
# print('queue[0][0]:', queue[0][0])
# print('queue[0][1]:', queue[0][1])
# print('queue[1][0]:', queue[1][0])
# print('queue[1][1]:', queue[1][1])
# print('-' * 100)
# ctx_norm = torch.arange(batch * seq_len * hidden_size).reshape(batch, seq_len, hidden_size)  # (batch x seq_len x h)
# print('ctx_norm:', ctx_norm)
# print('ctx_norm.size():', ctx_norm.size())
# ctx_norm = ctx_norm.unsqueeze(1)  # (batch x 1 x seq_len x hidden_size)
# print('ctx_norm:', ctx_norm)
# print('ctx_norm.size():', ctx_norm.size())
# ctx_norm = ctx_norm.repeat([1, K, 1, 1])  # (batch x K<复制> x seq_len x hidden_size)
# print('ctx_norm:', ctx_norm)
# print('ctx_norm.size():', ctx_norm.size())
# print('ctx_norm[:,0,:,:]:', ctx_norm[:, 0, :, :])
# print('ctx_norm[:,1,:,:]:', ctx_norm[:, 1, :, :])
# print('ctx_norm[0][0]:', ctx_norm[0][0])
# print('ctx_norm[0][1]:', ctx_norm[0][1])
# print('ctx_norm[1][0]:', ctx_norm[1][0])
# print('ctx_norm[1][1]:', ctx_norm[1][1])
# print('-' * 100)
# # self.queue：(batch_size<复制> x K x hidden_size x episode_len)
# # neg_A：(batch_size x K x seq_len x episode_len)
# neg_A = torch.einsum('bksh,bkhe->bkse', [ctx_norm, queue])
# print('neg_A[0][0]:', neg_A[0][0])
# print('neg_A[0][1]:', neg_A[0][1])
# print('neg_A[1][0]:', neg_A[1][0])
# print('neg_A[1][1]:', neg_A[1][1])

# """余弦相似度的计算"""
# batch = 3
# feature_size = 4
# feature = torch.arange(batch * feature_size).reshape(batch, feature_size)
# print('feature:', feature)
# vd_f_t = torch.arange(batch * feature_size).reshape(batch, feature_size)
# print('vd_f_t:', vd_f_t)
# re_loss = torch.einsum('bf,bf->b', [feature, vd_f_t]).unsqueeze(-1)  # (batch x 1)
# abs_feature = torch.sqrt(torch.einsum('bf,bf->b', [feature, feature])).unsqueeze(-1)
# abs_vd_f_t = torch.sqrt(torch.einsum('bf,bf->b', [vd_f_t, vd_f_t])).unsqueeze(-1)
# abs = torch.einsum('bj,bj->bj', [abs_feature, abs_vd_f_t])
# print('re_loss:', re_loss)
# print('re_loss.size():', re_loss.size())
# print('abs_feature:', abs_feature)
# print('abs_vd_f_t:', abs_vd_f_t)
# print('abs:', abs)
# print('cos:', re_loss / abs)

# """LSTMCell输出值的数值范围"""
# input = torch.tensor([2, 2, 2, 2], dtype=torch.float32).reshape(1, 4)
# h0 = torch.zeros((1, 3))
# c0 = torch.zeros((1, 3))
# lstm = nn.LSTMCell(4, 3)
# output = lstm(input, (h0, c0))
# print(output)

# """nn.Tanh()输入输出维度"""
# torch.set_printoptions(sci_mode=False)
# # input = torch.arange(2 * 4).reshape(2, 4)
# input = torch.tensor([[-1, -2, 3, -4],
#                       [5, 6, 7, 8]], dtype=torch.float32)
# sm = nn.Softmax(dim=-1)
# output = sm(input)
# print(output)

"""bias并不能决定nn.Linear()的输出值一定大于0"""
epoch = 1000
criterion = nn.MSELoss(reduction='mean')
target = torch.randn((3, 4))
input = torch.randn((3, 4), requires_grad=True)
fc = nn.Sequential(
    nn.Linear(4, 4, bias=True),
    nn.Linear(4, 4, bias=True)
)
optimizer = optim.SGD(fc.parameters(), lr=0.01)
for _ in range(epoch):
    optimizer.zero_grad()
    output = fc(input)
    loss = criterion(output, target)
    # """测试余弦相似度可不可以作为损失函数"""
    # cos = torch.einsum('bf,bf->b', [output, target]).unsqueeze(-1)  # (b x 1)
    # abs_o = torch.sqrt(torch.einsum('bf,bf->b', [output, output])).unsqueeze(-1)  # (b x 1)
    # abs_t = torch.sqrt(torch.einsum('bf,bf->b', [target, target])).unsqueeze(-1)  # (b x 1)
    # abs = torch.einsum('bf,bf->bf', [abs_o, abs_t])  # (b x 1)
    # cos = cos / abs
    # loss = ((-cos) / 2 + 0.5) * 10
    # print('cos:', cos.item())
    print('loss:', loss.item())
    loss.backward()
    optimizer.step()
    print('input:', input)
    print('target:', target)
    print('output:', output)

# """验证余弦相似度的物理意义，只能方向相似，不能长度相似"""
# target = torch.tensor([-2, -9, 8, 3], dtype=torch.float32).reshape(1, -1)
# output = torch.tensor([-0.1217, -0.5475, 0.4867, 0.1825], dtype=torch.float32).reshape(1, -1)
# cos = torch.einsum('bf,bf->b', [output, target]).unsqueeze(-1)  # (b x 1)
# print('cos:', cos)
# abs_o = torch.sqrt(torch.einsum('bf,bf->b', [output, output])).unsqueeze(-1)  # (b x 1)
# print('abs_o:', abs_o)
# abs_t = torch.sqrt(torch.einsum('bf,bf->b', [target, target])).unsqueeze(-1)  # (b x 1)
# print('abs_t:', abs_t)
# abs = torch.einsum('bf,bf->bf', [abs_o, abs_t])  # (b x 1)
# print('abs:', abs)
# cos = cos / abs
# print('target.size():', target.size())
# print('cos:', cos)

# """weighted_context的数值范围可能小于0也可能大于0"""
# batch = 2
# h_dim = 4
# v_dim = 4
# dot_dim = 3
# h = torch.tensor([[-2, -1, 2, 3],
#                   [-4, 2, 3, -3]], dtype=torch.float32)
# visual_context = torch.tensor([[[-1, 4, -3, 2],
#                                 [-2, -3, 4, -1]],
#                                [[-2, -3, 4, -1],
#                                 [-1, 4, -3, 2]]], dtype=torch.float32)
# linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
# linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
# sm = nn.Softmax(dim=1)
# target = linear_in_h(h).unsqueeze(2)  # (batch x dot_dim x 1)
# context = linear_in_v(visual_context)  # (batch x v_num x dot_dim)
# # Get attention
# attn = torch.bmm(context, target).squeeze(2)  # (batch x v_num)，得到36个视点的attention
# attn = sm(attn)  # TODO 计算出36个全景图像对预测action的贡献权重
# print('attn:', attn)
# attn3 = attn.view(attn.size(0), 1, attn.size(1))  # (batch x 1 x v_num)
# # TODO 计算全景图像的attention
# weighted_context = torch.bmm(
#     attn3, visual_context).squeeze(1)  # (batch x v_dim),TODO 数值可能小于0也可能大于0
# print('weighted_context:', weighted_context)

# """将[-1,1]的值投影到[0,2]"""
# x = torch.rand(10) * 2 - 1
# x = x.reshape(x.size(0), 1)
# print('x:', x)
# print('x.size():', x.size())
# y = (x / 2 + 0.5) * 10
# print('y:', y)
# print('y.size():', y.size())
