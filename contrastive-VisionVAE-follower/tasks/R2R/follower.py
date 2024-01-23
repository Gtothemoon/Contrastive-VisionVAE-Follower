""" Agents: stop/random/shortest/seq2seq  """

import json
import math
import sys

import einops
import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D  # 概率分布

from utils import vocab_pad_idx, vocab_eos_idx, flatten, structured_map, try_cuda

# from env import FOLLOWER_MODEL_ACTIONS, FOLLOWER_ENV_ACTIONS, IGNORE_ACTION_INDEX, LEFT_ACTION_INDEX, RIGHT_ACTION_INDEX, START_ACTION_INDEX, END_ACTION_INDEX, FORWARD_ACTION_INDEX, index_action_tuple

InferenceState = namedtuple("InferenceState",
                            "prev_inference_state, world_state, observation, flat_index, last_action, last_action_embedding, action_count, score, h_t, c_t, last_alpha")

Cons = namedtuple("Cons", "first, rest")


def cons_to_list(cons):
    l = []
    while True:
        l.append(cons.first)
        cons = cons.rest
        if cons is None:
            break
    return l


def backchain_inference_states(last_inference_state):
    states = []
    observations = []
    actions = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    attentions = []
    while inf_state is not None:
        states.append(inf_state.world_state)
        observations.append(inf_state.observation)
        actions.append(inf_state.last_action)
        attentions.append(inf_state.last_alpha)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(states)), list(reversed(observations)), list(reversed(actions))[1:], list(reversed(scores))[
                                                                                              1:], list(
        reversed(attentions))[1:]  # exclude start action


def least_common_viewpoint_path(inf_state_a, inf_state_b):
    # return inference states traversing from A to X, then from Y to B,
    # where X and Y are the least common ancestors of A and B respectively that share a viewpointId
    path_to_b_by_viewpoint = {}
    b = inf_state_b
    b_stack = Cons(b, None)
    while b is not None:
        path_to_b_by_viewpoint[b.world_state.viewpointId] = b_stack
        b = b.prev_inference_state
        b_stack = Cons(b, b_stack)
    a = inf_state_a
    path_from_a = [a]
    while a is not None:
        vp = a.world_state.viewpointId
        if vp in path_to_b_by_viewpoint:
            path_to_b = cons_to_list(path_to_b_by_viewpoint[vp])
            assert path_from_a[-1].world_state.viewpointId == path_to_b[0].world_state.viewpointId
            return path_from_a + path_to_b[1:]
        a = a.prev_inference_state
        path_from_a.append(a)
    raise AssertionError("no common ancestor found")


# TODO 构造句子batch（用于送入encoder）
def batch_instructions_from_encoded(encoded_instructions, max_length, reverse=False, sort=False):  # reverse=True
    # encoded_instructions: list of lists of token indices (should not be padded, or contain BOS or EOS tokens)
    # seq_tensor = np.array(encoded_instructions)
    # make sure pad does not start any sentence
    num_instructions = len(encoded_instructions)
    seq_tensor = np.full((num_instructions, max_length), vocab_pad_idx)  # (batch_size x max_len)全0矩阵
    seq_lengths = []
    for i, inst in enumerate(encoded_instructions):
        if len(inst) > 0:
            assert inst[-1] != vocab_eos_idx  # 到目前还没给句子加<EOS>
        if reverse:
            inst = inst[::-1]  # TODO 反转数组的巧妙操作
        inst = np.concatenate((inst, [vocab_eos_idx]))  # TODO 到这里才给句子加<EOS>
        inst = inst[:max_length]  # TODO 把数组的所有索引取到
        seq_tensor[i, :len(inst)] = inst
        seq_lengths.append(len(inst))

    seq_tensor = torch.from_numpy(seq_tensor)  # (batch_size x max_len)
    if sort:  # sort=True
        seq_lengths, perm_idx = torch.from_numpy(np.array(seq_lengths)).sort(0, True)  # 按句子长度从大到小排序
        seq_lengths = list(seq_lengths)
        seq_tensor = seq_tensor[perm_idx]  # 按句子长度从大到小排序

    mask = (seq_tensor == vocab_pad_idx)[:, :max(seq_lengths)]  # True/False张量

    ret_tp = try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
             try_cuda(mask.byte()), \
             seq_lengths
    if sort:
        ret_tp = ret_tp + (list(perm_idx),)
    return ret_tp  # seq_tensor，mask，seq_lengths，perm_idx


class BaseAgent(object):
    """ Base class for an R2R agent to generate and save trajectories. """

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = []  # For learning agents

    def write_results(self):
        results = {}
        for key, item in self.results.items():
            results[key] = {
                'instr_id': item['instr_id'],
                'trajectory': item['trajectory'],
            }
        with open(self.results_path, 'w') as f:
            json.dump(results, f)

    def rollout(self):
        """ Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  """
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def test(self):
        self.env.reset_epoch()  # ix = 0，重头加载数据的每个batch
        self.results = {}
        self.losses = []

        # We rely on env showing the entire batch before repeating anything
        # print 'Testing %s' % self.__class__.__name__
        looped = False
        rollout_scores = []
        beam_10_scores = []
        while True:
            rollout_results = self.rollout()  # TODO data完整流过model一次
            # if self.feedback == 'argmax':
            # beam_results = self.beam_search(1, load_next_minibatch=False)
            # assert len(rollout_results) == len(beam_results)
            # for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #     assert rollout_traj['instr_id'] == beam_trajs[0]['instr_id']
            #     assert rollout_traj['trajectory'] == beam_trajs[0]['trajectory']
            #     assert np.allclose(rollout_traj['score'], beam_trajs[0]['score'])
            # print("passed check: beam_search with beam_size=1")
            #
            # self.env.set_beam_size(10)
            # beam_results = self.beam_search(10, load_next_minibatch=False)
            # assert len(rollout_results) == len(beam_results)
            # for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #     rollout_score = rollout_traj['score']
            #     rollout_scores.append(rollout_score)
            #     beam_score = beam_trajs[0]['score']
            #     beam_10_scores.append(beam_score)
            #     # assert rollout_score <= beam_score
            # self.env.set_beam_size(1)
            # # print("passed check: beam_search with beam_size=10")
            # if self.feedback == 'teacher' and self.beam_size == 1:
            #     rollout_loss = self.loss
            #     path_obs, path_actions, encoded_instructions = self.env.gold_obs_actions_and_instructions(self.episode_len, load_next_minibatch=False)
            #     for i in range(len(rollout_results)):
            #         assert rollout_results[i]['actions'] == path_actions[i]
            #         assert [o1['viewpoint'] == o2['viewpoint']
            #                 for o1, o2 in zip(rollout_results[i]['observations'], path_obs[i])]
            #     trajs, loss = self._score_obs_actions_and_instructions(path_obs, path_actions, encoded_instructions)
            #     for traj, rollout in zip(trajs, rollout_results):
            #         assert traj['instr_id'] == rollout['instr_id']
            #         assert traj['actions'] == rollout['actions']
            #         assert np.allclose(traj['score'], rollout['score'])
            #     assert np.allclose(rollout_loss.data[0], loss.data[0])
            #     print('passed score test')

            for result in rollout_results:
                if result['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[result['instr_id']] = result
            if looped:
                break
        # if self.feedback == 'argmax':
        #     print("avg rollout score: ", np.mean(rollout_scores))
        #     print("avg beam 10 score: ", np.mean(beam_10_scores))
        return self.results


def path_element_from_observation(ob):
    return (ob['viewpoint'], ob['heading'], ob['elevation'])


class StopAgent(BaseAgent):
    """ An agent that doesn't move! """

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)]
        } for ob in obs]
        return traj


class RandomAgent(BaseAgent):
    """ An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. """

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)]
        } for ob in obs]
        ended = [False] * len(obs)

        self.steps = [0] * len(obs)
        for t in range(6):
            actions = []
            for i, ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append(0)  # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] == 0:
                    a = np.random.randint(len(ob['adj_loc_list']) - 1) + 1
                    actions.append(a)  # choose a random adjacent loc
                    self.steps[i] += 1
                else:
                    assert len(ob['adj_loc_list']) > 1
                    actions.append(1)  # go forward
                    self.steps[i] += 1
            world_states = self.env.step(world_states, actions, obs)
            obs = self.env.observe(world_states)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
        return traj


class ShortestAgent(BaseAgent):
    """ An agent that always takes the shortest path to goal. """

    def rollout(self):
        world_states = self.env.reset()
        # obs = self.env.observe(world_states)
        all_obs, all_actions = self.env.shortest_paths_to_goals(world_states, 20)
        return [
            {
                'instr_id': obs[0]['instr_id'],
                # end state will appear twice because stop action is a no-op, so exclude it
                'trajectory': [path_element_from_observation(ob) for ob in obs[:-1]]
            }
            for obs in all_obs
        ]


# TODO 核心agent
class Seq2SeqAgent(BaseAgent):
    """ An agent based on an LSTM seq2seq model with attention. """

    # TODO 高级action
    # For now, the agent can't pick which forward move to make - just the one in the middle
    # env_actions = FOLLOWER_ENV_ACTIONS
    # start_index = START_ACTION_INDEX
    # ignore_index = IGNORE_ACTION_INDEX
    # forward_index = FORWARD_ACTION_INDEX
    # end_index = END_ACTION_INDEX
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, v_encoder, v_decoder, episode_len=10, beam_size=1,
                 reverse_instruction=True,
                 max_instruction_length=80, beta=0.5, batch_size=100, K=1000, hidden_size=512):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.v_encoder = v_encoder  # 对图像向量降维，用于对比学习
        self.v_decoder = v_decoder  # 还原图像向量
        self.episode_len = episode_len
        self.losses = []  # for learning agent
        # 在计算CrossEntropyLoss时，真实的label（一个标量）被处理成onehot编码的形式
        # self.criterion = nn.CrossEntropyLoss(ignore_index=-1)  # TODO 对比损失要不要用同一个损失函数？
        self.criterion = try_cuda(nn.CrossEntropyLoss(ignore_index=-1))

        self.beam_size = beam_size
        self.reverse_instruction = reverse_instruction
        self.max_instruction_length = max_instruction_length

        # self.contrast_criterion = nn.CrossEntropyLoss()  # TODO 对比学习损失函数
        self.contrast_criterion = try_cuda(nn.CrossEntropyLoss())
        # self.re_criterion = nn.MSELoss()  # TODO 重建图像向量损失函数
        self.re_criterion = try_cuda(nn.MSELoss())
        self.sm = nn.Softmax(dim=-1)  # 沿着hidden_size做softmax
        self.beta = beta  # 递增loss超参数

        # TODO create the queue（是放在decoder还是agent？）
        self.K = K
        self.hidden_size = hidden_size
        self.queue = torch.randn(self.hidden_size, self.K, requires_grad=False)  # 负样本的个数K=1000
        self.queue = try_cuda(self.queue)  # TODO 提前把queue放到cuda上！
        # nn.functional.normalize()将某一个维度除以那个维度对应的范数(默认是2范数)
        # TODO 经过L2正则化后的向量点积就是这两个向量的余弦相似度，因为L2正则化帮我们对每个向量除以了分母的平方和!
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_ptr = torch.zeros(1, dtype=torch.long, requires_grad=False)  # 指针

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]  # keys: (batch x hidden_size)

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # queue：(hidden_size x K)
        # keys：(batch x hidden_size)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer，循环指针

        self.queue_ptr[0] = ptr

    # @staticmethod
    # def n_inputs():
    #     return len(FOLLOWER_MODEL_ACTIONS)
    #
    # @staticmethod
    # def n_outputs():
    #     return len(FOLLOWER_MODEL_ACTIONS)-2 # Model doesn't output start or ignore

    # TODO 提取一个batch上的所有图像特征
    def _feature_variables(self, obs, beamed=False):
        """ Extract precomputed features into variable. """
        # TODO ob['feature']是tuple类型
        feature_lists = list(zip(*[ob['feature'] for ob in (flatten(obs) if beamed else obs)]))  # [(batch_size)*36]?
        assert len(feature_lists) == len(self.env.image_features_list)  # =1
        batched = []
        for featurizer, feature_list in zip(self.env.image_features_list, feature_lists):
            batched.append(featurizer.batch_features(feature_list))
        return batched

    # TODO 一个batch的所有可导航方向特征
    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)  # TODO 相当于mask，(batch_size x max_num_a)
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros(
            (len(obs), max_num_a, action_embedding_dim),
            dtype=np.float32)  # (batch_size x max_num_a x embedding_dim)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']  # TODO 当前视点的所有navigations
            num_a = len(adj_loc_list)
            is_valid[i, 0:num_a] = 1.
            for n_a, adj_dict in enumerate(adj_loc_list):  # TODO 为什么还循环一遍？？？
                action_embeddings[i, :num_a, :] = ob['action_embedding']
        return (
            Variable(torch.from_numpy(action_embeddings), requires_grad=False).cuda(),
            Variable(torch.from_numpy(is_valid), requires_grad=False).cuda(),
            is_valid)

    # TODO 一个batch当前obs的真实action（相当于label）
    def _teacher_action(self, obs, ended):
        """ Extract teacher actions into variable. """
        a = torch.LongTensor(len(obs))  # (batch)
        for i, ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            a[i] = ob['teacher'] if not ended[i] else -1  # 如果a=-1，则计算loss的时候省略掉（因为已经预测结束）
        return try_cuda(Variable(a, requires_grad=False))

    # TODO 构造句子batch
    def _proc_batch(self, obs, beamed=False):
        encoded_instructions = [ob['instr_encoding'] for ob in (flatten(obs) if beamed else obs)]
        return batch_instructions_from_encoded(encoded_instructions, self.max_instruction_length,
                                               reverse=self.reverse_instruction)

    def rollout(self):
        if self.beam_size == 1:
            return self._rollout_with_loss()  # TODO 训练时是这个
        else:
            assert self.beam_size >= 1
            beams, _, _ = self.beam_search(self.beam_size)
            return [beam[0] for beam in beams]

    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions):  # TODO 没用到
        batch_size = len(path_obs)
        assert len(path_actions) == batch_size
        assert len(encoded_instructions) == batch_size
        for path_o, path_a in zip(path_obs, path_actions):
            assert len(path_o) == len(path_a) + 1

        # TODO nlp模型常规操作
        seq, seq_mask, seq_lengths, perm_indices = \
            batch_instructions_from_encoded(
                encoded_instructions, self.max_instruction_length,
                reverse=self.reverse_instruction, sort=True)
        loss = 0

        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        u_t_prev = self.decoder.u_begin.expand(batch_size, -1)  # init action
        ended = np.array([False] * batch_size)
        sequence_scores = try_cuda(torch.zeros(batch_size))

        # TODO 初始化预测轨迹
        traj = [{
            'instr_id': path_o[0]['instr_id'],
            'trajectory': [path_element_from_observation(path_o[0])],
            'actions': [],
            'scores': [],
            'observations': [path_o[0]],
            'instr_encoding': path_o[0]['instr_encoding']
        } for path_o in path_obs]

        obs = None
        for t in range(self.episode_len):  # TODO 不懂
            next_obs = []
            next_target_list = []
            for perm_index, src_index in enumerate(perm_indices):
                path_o = path_obs[src_index]
                path_a = path_actions[src_index]
                if t < len(path_a):
                    next_target_list.append(path_a[t])
                    next_obs.append(path_o[t])
                else:
                    next_target_list.append(-1)
                    next_obs.append(obs[perm_index])

            obs = next_obs

            target = try_cuda(Variable(torch.LongTensor(next_target_list), requires_grad=False))

            f_t_list = self._feature_variables(obs)  # Image features from obs
            all_u_t, is_valid, _ = self._action_variable(obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx, seq_mask)

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')

            # Supervised training
            loss += self.criterion(logit, target)

            # Determine next model inputs
            a_t = torch.clamp(target, min=0)  # teacher forcing
            # update the previous action
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()  # TODO 从中选一

            action_scores = -F.cross_entropy(logit, target, ignore_index=-1, reduce=False).data
            sequence_scores += action_scores

            # Save trajectory output
            for perm_index, src_index in enumerate(perm_indices):
                ob = obs[perm_index]
                if not ended[perm_index]:
                    traj[src_index]['trajectory'].append(path_element_from_observation(ob))
                    traj[src_index]['score'] = float(sequence_scores[perm_index])
                    traj[src_index]['scores'].append(action_scores[perm_index])
                    traj[src_index]['actions'].append(a_t.data[perm_index])
                    # traj[src_index]['observations'].append(ob)

            # Update ended list
            for i in range(batch_size):
                action_idx = a_t[i].data[0]
                if action_idx == 0:
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        return traj, loss

    # 完整的训练一个batch
    def _rollout_with_loss(self):
        initial_world_states = self.env.reset(sort=True)  # 获取下一个batch的初始states，例，(scanId, viewpointId, heading, 0)
        initial_obs = self.env.observe(initial_world_states)
        initial_obs = np.array(initial_obs)  # 从环境中得到当前batch的obs信息
        batch_size = len(initial_obs)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(initial_obs)  # 构造句子batch

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        # TODO consider not feeding this into the decoder, and just using attention

        self.loss = 0  # 一个batch所有时间步的loss，大于0
        self.pos_scores = 0  # 正例的scores
        self.neg_scores = 0  # 负例的scores
        self.InfoNCE = 0  # 对比学习损失，大于0
        self.VAE_loss = 0  # VAE损失
        self.re_loss = 0  # 图像向量重建损失，必须大于0
        self.d_KL = 0  # KL Divergence

        feedback = self.feedback
        # ctx：(batch x seq_len x hidden_size*num_directions)，  # hidden_size*num_directions = 512
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)  # 将句子信息输入encoder

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)],
            'actions': [],
            'scores': [],
            'observations': [ob],
            'instr_encoding': ob['instr_encoding']
        } for ob in initial_obs]

        obs = initial_obs
        world_states = initial_world_states

        # Initial action
        # u_t: next navigable direction
        u_t_prev = self.decoder.u_begin.expand(batch_size, -1)  # (batch_size x action_embedding_size)
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        env_action = [None] * batch_size
        sequence_scores = try_cuda(torch.zeros(batch_size))

        contrast_feat = torch.zeros((batch_size, self.episode_len, self.hidden_size))  # 图像embedding序列，用于对比学习
        contrast_feat = try_cuda(contrast_feat)  # TODO 记得放到gpu上！

        ve_h_t, ve_c_t = self.v_encoder.init_state(batch_size)  # v_encoder 隐藏状态初始化
        vd_f_t, vd_c_t = self.v_decoder.init_state(batch_size)  # v_decoder 隐藏状态初始化

        # decoder翻译action
        for t in range(self.episode_len):
            # Image features from obs
            f_t_list = self._feature_variables(obs)  # v_t，全景图像，数值可能小于0也可能大于0
            # all_u_t是action_embeddings，(batch_size x max_num_a x embedding_dim)
            all_u_t, is_valid, _ = self._action_variable(obs)  # u_t，actions

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'  # len(f_t_list)必须==1

            # (batch x feature_size)
            feature, ve_h_t, ve_h_t_drop, ve_c_t, alpha_v, sampled_z, mu, log_sigma = \
                self.v_encoder(f_t_list[0], ve_h_t, ve_c_t)  # TODO 包含VAE encoder
            contrast_feat[:, t, :] = ve_h_t_drop  # 保存图像序列，用于对比学习(直接tensor保存)
            re_feature, vd_f_t, vd_c_t = self.v_decoder(sampled_z, vd_f_t, vd_c_t)  # TODO 包含VAE decoder

            self.re_loss = self.re_criterion(feature, re_feature)  # 图像重建损失
            # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
            self.d_KL = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1. - log_sigma)  # 最小化
            self.VAE_loss += (self.re_loss + self.d_KL)

            h_t, c_t, alpha, logit = self.decoder(
                u_t_prev, all_u_t, re_feature, h_t, c_t, ctx, seq_mask)

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')  # 注意：不需要softmax，直接输入进CrossEntropyLoss

            # Supervised training
            target = self._teacher_action(obs, ended)  # (batch_size x 1)
            # TODO 改动的想法：越接近目的地，loss的权重越大，避免偏差累积？
            gamma = math.sqrt((self.episode_len + self.beta * t) / self.episode_len)  # 递增loss的增量超参数
            self.loss += gamma * self.criterion(logit, target)  # 计算当前时间步的action的递增loss

            # Determine next model inputs
            if feedback == 'teacher':
                # turn -1 (ignore) to 0 (stop) so that the action is executable
                # TODO torch.clamp()把张量中每个值裁剪到给定范围内
                a_t = torch.clamp(target, min=0)  # 相当于把所有等于-1的值变成0
            elif feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()  # 预测出来的a_t不会等于-1，所以不用clamp()
            elif feedback == 'sample':
                probs = F.softmax(logit, dim=1)  # sampling an action from model
                # Further mask probs where agent can't move forward
                # Note input to `D.Categorical` does not have to sum up to 1
                # http://pytorch.org/docs/stable/torch.html#torch.multinomial
                probs[is_valid == 0] = 0.
                m = D.Categorical(probs)
                a_t = m.sample()
            else:
                sys.exit('Invalid feedback option')

            # update the previous action
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()

            action_scores = -F.cross_entropy(logit, a_t, ignore_index=-1, reduce=False).data  # 一个batch当前时间步的action分数
            sequence_scores += action_scores  # 一个batch所有时间步的action分数

            # dfried: I changed this so that the ended list is updated afterward; this causes <end> to be added as the last action, along with its score, and the final world state will be duplicated (to more closely match beam search)
            # Make environment action
            for i in range(batch_size):
                # action_idx = a_t[i].data[0]  # 报错，改用下一句
                action_idx = a_t[i].item()
                env_action[i] = action_idx

            world_states = self.env.step(world_states, env_action, obs)  # 更新环境状态
            obs = self.env.observe(world_states)  # 更新环境信息
            # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.data[0], sequence_scores[0]))

            # Save trajectory output
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
                    traj[i]['score'] = sequence_scores[i]
                    traj[i]['scores'].append(action_scores[i])
                    traj[i]['actions'].append(a_t.data[i])
                    traj[i]['observations'].append(ob)

            # Update ended list
            for i in range(batch_size):
                # action_idx = a_t[i].data[0]  # 报错，改用下一句
                action_idx = a_t[i].item()
                if action_idx == 0:  # a_t=0表示原地不动
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                # 提前结束翻译，说明path_len <= episode_len
                if t < self.episode_len - 1:
                    for p in range(t + 1, self.episode_len):
                        contrast_feat[:, p, :] = contrast_feat[:, t, :]  # 补全张量
                break

        # self.losses.append(self.loss.data[0] / self.episode_len)
        # shouldn't divide by the episode length because of masking

        """
        ctx: (batch x seq_len x hidden_size*num_directions)
        contrast_feat: (batch x episode_len x hidden_size)
        计算句子和图像序列的相似度，参考论文：Multi-modal Discriminative Model for Vision-and-Language Navigation
        采用MoCo的队列方法存储负例，为了减少gpu显存占用，先将时间维度降成1维，然后使用常规对比学习损失计算方法
        """
        o_q = torch.bmm(ctx.transpose(-2, -1), ctx)  # (batch x hidden_size x hidden_size)
        o_k = torch.bmm(contrast_feat.transpose(-2, -1), contrast_feat)  # (batch x hidden_size x hidden_size)
        weights_q = self.sm(o_q)  # (batch x hidden_size x hidden_size)
        weights_k = self.sm(o_k)  # (batch x hidden_size x hidden_size)
        q = torch.einsum('bhz,bhz->bh', [weights_q, o_q])  # (batch x hidden_size)
        k = torch.einsum('bhz,bhz->bh', [weights_k, o_k])  # (batch x hidden_size)
        q = nn.functional.normalize(q, dim=-1)  # (batch x hidden_size)
        k = nn.functional.normalize(k, dim=-1)  # (batch x hidden_size)
        # 计算正例scores
        self.pos_scores = torch.einsum('bh,bh->b', [q, k]).unsqueeze(-1)  # (batch x 1)
        # 计算负例scores
        self.neg_scores = torch.einsum('bh,hk->bk', [q, self.queue.clone().detach()])  # (batch x K)
        total_scores = torch.cat([self.pos_scores, self.neg_scores], dim=1)  # (batch x 1+K)
        contrast_labels = try_cuda(torch.zeros(total_scores.shape[0], dtype=torch.long))  # 正例永远在索引值=0处
        self.InfoNCE = self.contrast_criterion(total_scores, contrast_labels)  # 对比学习损失InfoNCE
        # 负例入队出队
        self._dequeue_and_enqueue(k)

        # TODO ↓测试用：三个loss的数量级不能差太大，验证InfoNCE和VAE_loss的数值范围
        # print('print: loss:{},InfoNCE:{},VAE_loss:{}'.format(self.loss, self.InfoNCE, self.VAE_loss))
        # print('print: pos_scores:{}'.format(self.pos_scores))
        # print('print: neg_scores:{}'.format(self.neg_scores))

        self.loss = self.loss + self.VAE_loss + self.InfoNCE  # 总loss = 预测动作loss + VAE_loss + 对比学习loss
        # self.losses.append(self.loss.data[0])  # 源代码这句应该会报错
        self.losses.append(self.loss.item())  # 改用item()获取loss的数值

        return traj  # 返回预测轨迹

    def beam_search(self, beam_size, load_next_minibatch=True, mask_undo=False):  # TODO 这里没用
        assert self.env.beam_size >= beam_size
        world_states = self.env.reset(sort=True, beamed=True, load_next_minibatch=load_next_minibatch)
        obs = self.env.observe(world_states, beamed=True)
        batch_size = len(world_states)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(obs, beamed=True)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)

        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [
            [InferenceState(prev_inference_state=None,
                            world_state=ws[0],
                            observation=o[0],
                            flat_index=i,
                            last_action=-1,
                            last_action_embedding=self.decoder.u_begin.view(-1),
                            action_count=0,
                            score=0.0, h_t=None, c_t=None, last_alpha=None)]
            for i, (ws, o) in enumerate(zip(world_states, obs))
        ]

        # Do a sequence rollout and calculate the loss
        for t in range(self.episode_len):
            flat_indices = []
            beam_indices = []
            u_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    u_t_list.append(inf_state.last_action_embedding)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            flat_obs = flatten(obs)
            f_t_list = self._feature_variables(flat_obs)  # Image features from obs
            all_u_t, is_valid, is_valid_numpy = self._action_variable(flat_obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t[flat_indices], c_t[flat_indices], ctx[beam_indices],
                seq_mask[beam_indices])

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            if mask_undo:
                masked_logit = logit.clone()
            else:
                masked_logit = logit

            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            # action_scores, action_indices = log_probs.topk(min(beam_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            new_beams = []
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states, beam_obs) in enumerate(zip(beams, world_states, obs)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                assert len(beam_obs) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, ob, action_score_row, action_index_row) in \
                            enumerate(zip(beam, beam_world_states, beam_obs, action_scores[start_index:end_index],
                                          action_indices[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_score, action_index in zip(action_score_row, action_index_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                               world_state=world_state,
                                               # will be updated later after successors are pruned
                                               observation=ob,  # will be updated later after successors are pruned
                                               flat_index=flat_index,
                                               last_action=action_index,
                                               last_action_embedding=all_u_t[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=float(inf_state.score + action_score), h_t=None, c_t=None,
                                               last_alpha=alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_world_states, successor_env_actions, successor_last_obs,
                                                   beamed=True)
            successor_obs = self.env.observe(successor_world_states, beamed=True)

            all_successors = structured_map(
                lambda inf_state, world_state, obs: inf_state._replace(world_state=world_state, observation=obs),
                all_successors, successor_world_states, successor_obs, nested=True)

            # if all_successors[0]:
            #     print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, all_successors[0][0].world_state, all_successors[0][0].last_action, all_successors[0][0].score))

            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_action == 0 or t == self.episode_len - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]

            obs = [
                [inf_state.observation for inf_state in beam]
                for beam in beams
            ]

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

        trajs = []

        for this_completed in completed:
            assert this_completed
            this_trajs = []
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(
                    inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'trajectory': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        traversed_lists = None  # todo
        return trajs, completed, traversed_lists

    def state_factored_search(self, completion_size, successor_size, load_next_minibatch=True, mask_undo=False,
                              first_n_ws_key=4):
        assert self.env.beam_size >= successor_size
        world_states = self.env.reset(sort=True, beamed=True, load_next_minibatch=load_next_minibatch)
        initial_obs = self.env.observe(world_states, beamed=True)
        batch_size = len(world_states)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(initial_obs, beamed=True)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)

        completed = []
        completed_holding = []
        for _ in range(batch_size):
            completed.append({})
            completed_holding.append({})

        state_cache = [
            {ws[0][0:first_n_ws_key]: (InferenceState(prev_inference_state=None,
                                                      world_state=ws[0],
                                                      observation=o[0],
                                                      flat_index=None,
                                                      last_action=-1,
                                                      last_action_embedding=self.decoder.u_begin.view(-1),
                                                      action_count=0,
                                                      score=0.0, h_t=h_t[i], c_t=c_t[i], last_alpha=None), True)}
            for i, (ws, o) in enumerate(zip(world_states, initial_obs))
        ]

        beams = [[inf_state for world_state, (inf_state, expanded) in sorted(instance_cache.items())]
                 for instance_cache in
                 state_cache]  # sorting is a noop here since each instance_cache should only contain one

        # traversed_lists = None
        # list of inference states containing states in order of the states being expanded
        last_expanded_list = []
        traversed_lists = []
        for beam in beams:
            assert len(beam) == 1
            first_state = beam[0]
            last_expanded_list.append(first_state)
            traversed_lists.append([first_state])

        def update_traversed_lists(new_visited_inf_states):
            assert len(new_visited_inf_states) == len(last_expanded_list)
            assert len(new_visited_inf_states) == len(traversed_lists)

            for instance_index, instance_states in enumerate(new_visited_inf_states):
                last_expanded = last_expanded_list[instance_index]
                # todo: if this passes, shouldn't need traversed_lists
                assert last_expanded.world_state.viewpointId == traversed_lists[instance_index][
                    -1].world_state.viewpointId
                for inf_state in instance_states:
                    path_from_last_to_next = least_common_viewpoint_path(last_expanded, inf_state)
                    # path_from_last should include last_expanded's world state as the first element, so check and drop that
                    assert path_from_last_to_next[0].world_state.viewpointId == last_expanded.world_state.viewpointId
                    assert path_from_last_to_next[-1].world_state.viewpointId == inf_state.world_state.viewpointId
                    traversed_lists[instance_index].extend(path_from_last_to_next[1:])
                    last_expanded = inf_state
                last_expanded_list[instance_index] = last_expanded

        # Do a sequence rollout and calculate the loss
        while any(len(comp) < completion_size for comp in completed):
            beam_indices = []
            u_t_list = []
            h_t_list = []
            c_t_list = []
            flat_obs = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    u_t_list.append(inf_state.last_action_embedding)
                    h_t_list.append(inf_state.h_t.unsqueeze(0))
                    c_t_list.append(inf_state.c_t.unsqueeze(0))
                    flat_obs.append(inf_state.observation)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            f_t_list = self._feature_variables(flat_obs)  # Image features from obs
            all_u_t, is_valid, is_valid_numpy = self._action_variable(flat_obs)
            h_t = torch.cat(h_t_list, dim=0)
            c_t = torch.cat(c_t_list, dim=0)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx[beam_indices], seq_mask[beam_indices])

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            if mask_undo:
                masked_logit = logit.clone()
            else:
                masked_logit = logit

            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            # _, action_indices = masked_logit.data.topk(min(successor_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(logit.size()[1], dim=1)  # todo: fix this
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states) in enumerate(zip(beams, world_states)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, action_score_row) in \
                            enumerate(zip(beam, beam_world_states, log_probs[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_index, action_score in enumerate(action_score_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                               world_state=world_state,
                                               # will be updated later after successors are pruned
                                               observation=flat_obs[flat_index],
                                               # will be updated later after successors are pruned
                                               flat_index=None,
                                               last_action=action_index,
                                               last_action_embedding=all_u_t[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=inf_state.score + action_score,
                                               h_t=h_t[flat_index], c_t=c_t[flat_index],
                                               last_alpha=alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_world_states, successor_env_actions, successor_last_obs,
                                                   beamed=True)

            all_successors = structured_map(lambda inf_state, world_state: inf_state._replace(world_state=world_state),
                                            all_successors, successor_world_states, nested=True)

            # if all_successors[0]:
            #     print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, all_successors[0][0].world_state, all_successors[0][0].last_action, all_successors[0][0].score))

            assert len(all_successors) == len(state_cache)

            new_beams = []

            for beam_index, (successors, instance_cache) in enumerate(zip(all_successors, state_cache)):
                # early stop if we've already built a sizable completion list
                instance_completed = completed[beam_index]
                instance_completed_holding = completed_holding[beam_index]
                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                    continue
                for successor in successors:
                    ws_keys = successor.world_state[0:first_n_ws_key]
                    if successor.last_action == 0 or successor.action_count == self.episode_len:
                        if ws_keys not in instance_completed_holding or instance_completed_holding[ws_keys][
                            0].score < successor.score:
                            instance_completed_holding[ws_keys] = (successor, False)
                    else:
                        if ws_keys not in instance_cache or instance_cache[ws_keys][0].score < successor.score:
                            instance_cache[ws_keys] = (successor, False)

                # third value: did this come from completed_holding?
                uncompleted_to_consider = ((ws_keys, inf_state, False) for (ws_keys, (inf_state, expanded)) in
                                           instance_cache.items() if not expanded)
                completed_to_consider = ((ws_keys, inf_state, True) for (ws_keys, (inf_state, expanded)) in
                                         instance_completed_holding.items() if not expanded)
                import itertools
                import heapq
                to_consider = itertools.chain(uncompleted_to_consider, completed_to_consider)
                ws_keys_and_inf_states = heapq.nlargest(successor_size, to_consider, key=lambda pair: pair[1].score)

                new_beam = []
                for ws_keys, inf_state, is_completed in ws_keys_and_inf_states:
                    if is_completed:
                        assert instance_completed_holding[ws_keys] == (inf_state, False)
                        instance_completed_holding[ws_keys] = (inf_state, True)
                        if ws_keys not in instance_completed or instance_completed[ws_keys].score < inf_state.score:
                            instance_completed[ws_keys] = inf_state
                    else:
                        instance_cache[ws_keys] = (inf_state, True)
                        new_beam.append(inf_state)

                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                else:
                    new_beams.append(new_beam)

            beams = new_beams

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]
            successor_obs = self.env.observe(world_states, beamed=True)
            beams = structured_map(lambda inf_state, obs: inf_state._replace(observation=obs),
                                   beams, successor_obs, nested=True)
            update_traversed_lists(beams)

        completed_list = []
        for this_completed in completed:
            completed_list.append(
                sorted(this_completed.values(), key=lambda t: t.score, reverse=True)[:completion_size])
        completed_ws = [
            [inf_state.world_state for inf_state in comp_l]
            for comp_l in completed_list
        ]
        completed_obs = self.env.observe(completed_ws, beamed=True)
        completed_list = structured_map(lambda inf_state, obs: inf_state._replace(observation=obs),
                                        completed_list, completed_obs, nested=True)
        # TODO: consider moving observations and this update earlier so that we don't have to traverse as far back
        update_traversed_lists(completed_list)

        # TODO: sanity check the traversed lists here

        trajs = []
        for this_completed in completed_list:
            assert this_completed
            this_trajs = []
            for inf_state in this_completed:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(
                    inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'trajectory': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        # completed_list: list of lists of final inference states corresponding to the candidates, one list per instance
        # traversed_lists: list of "physical states" that the robot has explored, one per instance
        return trajs, completed_list, traversed_lists

    def set_beam_size(self, beam_size):
        if self.env.beam_size < beam_size:
            self.env.set_beam_size(beam_size)
        self.beam_size = beam_size

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, beam_size=1):
        """ Evaluate once on each instruction in the current environment """
        if not allow_cheat:  # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample']  # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.v_encoder.train()
            self.v_decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.v_encoder.eval()
            self.v_decoder.eval()
        self.set_beam_size(beam_size)
        return super(Seq2SeqAgent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, v_encoder_optimizer, v_decoder_optimizer, n_iters,
              feedback='teacher'):  # n_iters=100
        """ Train for a given number of iterations """
        assert all(f in self.feedback_options for f in feedback.split("+"))
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.v_encoder.train()
        self.v_decoder.train()
        self.losses = []
        it = range(1, n_iters + 1)
        try:
            import tqdm  # tqdm 是一个快速，可扩展的 Python 进度条，可以在 Python 长循环中添加一个进度提示信息
            it = tqdm.tqdm(it)
        except:
            pass
        for _ in it:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            v_encoder_optimizer.zero_grad()
            v_decoder_optimizer.zero_grad()
            self._rollout_with_loss()
            self.loss.backward()  # 总loss = 预测动作loss + 重建loss + 对比学习loss
            encoder_optimizer.step()
            decoder_optimizer.step()
            v_encoder_optimizer.step()
            v_decoder_optimizer.step()

    def _encoder_and_decoder_paths(self, base_path):
        return base_path + "_enc", base_path + "_dec", base_path + "_v_enc", base_path + "_v_dec"

    def save(self, path):
        """ Snapshot models """
        encoder_path, decoder_path, v_encoder_path, v_decoder_path = self._encoder_and_decoder_paths(path)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        torch.save(self.v_encoder.state_dict(), v_encoder_path)
        torch.save(self.v_decoder.state_dict(), v_decoder_path)

    def load(self, path, **kwargs):
        """ Loads parameters (but not training state) """
        encoder_path, decoder_path, v_encoder_path, v_decoder_path = self._encoder_and_decoder_paths(path)
        self.encoder.load_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_state_dict(torch.load(decoder_path, **kwargs))
        self.v_encoder.load_state_dict(torch.load(v_encoder_path, **kwargs))
        self.v_decoder.load_state_dict(torch.load(v_decoder_path, **kwargs))
