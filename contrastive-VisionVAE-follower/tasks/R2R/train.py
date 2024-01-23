import torch
from torch import optim
import os
import os.path
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utils
from utils import read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda
from env import R2RBatch, ImageFeatures
from model import EncoderLSTM, VisionAttnEecoderLSTM, VisionDecoderLSTM, LangAttnDecoderLSTM
from follower import Seq2SeqAgent
import eval

from vocab import SUBTRAIN_VOCAB, TRAINVAL_VOCAB, TRAIN_VOCAB

RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'

# TODO: how much is this truncating instructions?
MAX_INPUT_LENGTH = 80  # 最长句子长度

BATCH_SIZE = 100
max_episode_len = 10
word_embedding_size = 300  # 词嵌入维度
glove_path = 'tasks/R2R/data/train_glove.npy'  # 文件有
action_embedding_size = 2048 + 128  # 论文里说的 u_i，高级action
hidden_size = 512
dropout_ratio = 0.5
# feedback_method = 'sample' # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
FEATURE_SIZE = 2048 + 128  # 论文里说的 v_i，当前视点的navigations
log_every = 100
save_every = 1000


# TODO 就是为了得到模型的唯一名称
def get_model_prefix(args, image_feature_list):
    image_feature_name = "+".join(
        [featurizer.get_name() for featurizer in image_feature_list])
    model_prefix = 'follower_{}_{}'.format(
        args.feedback_method, image_feature_name)  # follower_teacher_imagenet
    if args.use_train_subset:
        model_prefix = 'trainsub_' + model_prefix
    if args.bidirectional:
        model_prefix = model_prefix + "_bidirectional"
    if args.use_pretraining:
        model_prefix = model_prefix.replace(
            'follower', 'follower_with_pretraining', 1)  # follower_with_pretraining_teacher_imagenet
    return model_prefix


def eval_model(agent, results_path, use_dropout, feedback, allow_cheat=False):  # TODO 没用到
    agent.results_path = results_path
    agent.test(
        use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat)


# TODO 过滤模型中需要计算梯度的参数
def filter_param(param_list):
    return [p for p in param_list if p.requires_grad]


# TODO 完整训练 n_iters 次
def train(args, train_env, agent, encoder_optimizer, decoder_optimizer, v_encoder_optimizer, v_decoder_optimizer,
          n_iters, log_every=log_every, val_envs=None):  # 有val_envs
    """ Train on training set, validating on both seen and unseen. """

    if val_envs is None:
        val_envs = {}

    print('Training with %s feedback' % args.feedback_method)

    data_log = defaultdict(list)
    start = time.time()

    split_string = "-".join(train_env.splits)

    # TODO 模型信息保存路径
    def make_path(n_iter):
        return os.path.join(
            SNAPSHOT_DIR, '%s_%s_iter_%d' % (
                get_model_prefix(args, train_env.image_features_list),
                split_string, n_iter))

    best_metrics = {}
    last_model_saved = {}
    # 每个iter训练一个batch
    for idx in range(0, n_iters, log_every):  # 0,100,200,...
        agent.env = train_env  # 切换到训练环境

        interval = min(log_every, n_iters - idx)  # 大部分时间是100
        iter = idx + interval
        data_log['iteration'].append(iter)  # 数学上的操作，保证iter为[100,n_iters]

        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, v_encoder_optimizer, v_decoder_optimizer, interval,
                    feedback=args.feedback_method)

        print('training on train_dataset')

        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval  # 一个batch有100个样本
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        save_log = []
        # Run validation
        for env_name, (val_env, evaluator) in sorted(val_envs.items()):
            agent.env = val_env  # 切换到验证环境
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=args.feedback_method,
                       allow_cheat=True)  # 启用dropout
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)

            agent.results_path = '%s%s_%s_iter_%d.json' % (
                RESULT_DIR, get_model_prefix(
                    args, train_env.image_features_list),
                env_name, iter)  # 预测轨迹保存路径

            # Get validation distance from goal under evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')  # 禁止梯度回传
            if not args.no_save:
                agent.write_results()  # 保存结果数据，{'instr_id': k, 'trajectory': v}，路径为 agent.results_path
            print("evaluating on {}".format(env_name))
            score_summary, _ = evaluator.score_results(agent.results)  # TODO 在模拟器上获得对验证集结果的评分

            for metric, val in sorted(score_summary.items()):
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)

                    key = (env_name, metric)
                    if key not in best_metrics or best_metrics[key] < val:
                        best_metrics[key] = val
                        if not args.no_save:
                            model_path = make_path(iter) + "_%s-%s=%.3f" % (
                                env_name, metric, val)
                            save_log.append(
                                "new best, saved model to %s" % model_path)
                            agent.save(model_path)  # 保存成功率最高的model
                            if key in last_model_saved:
                                for old_model_path in agent._encoder_and_decoder_paths(last_model_saved[key]):
                                    os.remove(old_model_path)
                            last_model_saved[key] = model_path

        print(('%s (%d %d%%) %s' % (
            timeSince(start, float(iter) / n_iters),
            iter, float(iter) / n_iters * 100, loss_str)))
        for s in save_log:
            print(s)

        if not args.no_save:
            if save_every and iter % save_every == 0:
                agent.save(make_path(iter))

            df = pd.DataFrame(data_log)  # 类csv、excel文件格式
            df.set_index('iteration')  # 设置以哪个索引为第一列
            df_path = '%s%s_%s_log.csv' % (
                PLOT_DIR, get_model_prefix(
                    args, train_env.image_features_list), split_string)
            df.to_csv(df_path)  # 保存每个iteration的train_loss和val_loss


# TODO 结果复现
def setup():
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    torch.manual_seed(1)  # TODO 多做几次实验，不需要重复性
    torch.cuda.manual_seed(1)
    pass


def make_more_train_env(args, train_vocab_path, train_splits,
                        batch_size=BATCH_SIZE):
    setup()
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    train_env = R2RBatch(image_features_list, batch_size=batch_size,
                         splits=train_splits, tokenizer=tok)
    return train_env  # TODO 增强数据集环境


def make_env_and_models(args, train_vocab_path, train_splits, test_splits,
                        batch_size=BATCH_SIZE):
    setup()
    image_features_list = ImageFeatures.from_args(args)  # TODO 存放类的list，[MeanPooledImageFeatures]
    vocab = read_vocab(train_vocab_path)  # 获取所有单词
    tok = Tokenizer(vocab=vocab)  # 构造单词与索引一一对应的词典
    train_env = R2RBatch(image_features_list, batch_size=batch_size,
                         splits=train_splits, tokenizer=tok)

    enc_hidden_size = hidden_size // 2 if args.bidirectional else hidden_size  # 保证encoder的hidden_size不变
    glove = np.load(glove_path)
    feature_size = FEATURE_SIZE
    encoder = try_cuda(EncoderLSTM(
        len(vocab), word_embedding_size, enc_hidden_size, vocab_pad_idx,
        dropout_ratio, bidirectional=args.bidirectional, glove=glove))
    decoder = try_cuda(LangAttnDecoderLSTM(
        action_embedding_size, hidden_size, dropout_ratio,
        feature_size=feature_size))
    v_encoder = try_cuda(VisionAttnEecoderLSTM(hidden_size, dropout_ratio, feature_size=feature_size))
    v_decoder = try_cuda(VisionDecoderLSTM(hidden_size, dropout_ratio, feature_size=feature_size))
    test_envs = {
        split: (R2RBatch(image_features_list, batch_size=batch_size,
                         splits=[split], tokenizer=tok),
                eval.Evaluation([split]))
        for split in test_splits}

    return train_env, test_envs, encoder, decoder, v_encoder, v_decoder  # 训练集环境、验证集环境，模型模块


def train_setup(args, batch_size=BATCH_SIZE):
    train_splits = ['train']
    # val_splits = ['train_subset', 'val_seen', 'val_unseen']
    val_splits = ['val_seen', 'val_unseen']
    if args.use_test_set:
        val_splits.append('test')

    vocab = TRAIN_VOCAB  # 所有env都用这个词表

    if args.use_train_subset:
        train_splits = ['sub_' + split for split in train_splits]
        val_splits = ['sub_' + split for split in val_splits]
        vocab = SUBTRAIN_VOCAB

    train_env, val_envs, encoder, decoder, v_encoder, v_decoder = make_env_and_models(
        args, vocab, train_splits, val_splits, batch_size=batch_size)
    if args.use_pretraining:
        pretrain_splits = args.pretrain_splits
        assert len(pretrain_splits) > 0, \
            'must specify at least one pretrain split'
        pretrain_env = make_more_train_env(
            args, vocab, pretrain_splits, batch_size=batch_size)  # 增强数据集

    # TODO 核心模型
    agent = Seq2SeqAgent(
        train_env, "", encoder, decoder, v_encoder, v_decoder, max_episode_len,
        max_instruction_length=MAX_INPUT_LENGTH, batch_size=batch_size)  # 没有给保存results的路径

    if args.use_pretraining:
        return agent, train_env, val_envs, pretrain_env
    else:
        return agent, train_env, val_envs


# Test set prediction will be handled separately
# def test_setup(args, batch_size=BATCH_SIZE):
#     train_env, test_envs, encoder, decoder = make_env_and_models(
#         args, TRAINVAL_VOCAB, ['train', 'val_seen', 'val_unseen'], ['test'],
#         batch_size=batch_size)
#     agent = Seq2SeqAgent(
#         None, "", encoder, decoder, max_episode_len,
#         max_instruction_length=MAX_INPUT_LENGTH)
#     return agent, train_env, test_envs


def train_val(args):
    """ Train on the training set, and validate on seen and unseen splits. """
    if args.use_pretraining:
        agent, train_env, val_envs, pretrain_env = train_setup(args)
    else:
        agent, train_env, val_envs = train_setup(args)

    encoder_optimizer = optim.Adam(
        filter_param(agent.encoder.parameters()), lr=learning_rate,
        weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(
        filter_param(agent.decoder.parameters()), lr=learning_rate,
        weight_decay=weight_decay)
    v_encoder_optimizer = optim.Adam(
        filter_param(agent.v_encoder.parameters()), lr=learning_rate,
        weight_decay=weight_decay)
    v_decoder_optimizer = optim.Adam(
        filter_param(agent.v_decoder.parameters()), lr=learning_rate,
        weight_decay=weight_decay)

    if args.use_pretraining:
        train(args, pretrain_env, agent, encoder_optimizer, decoder_optimizer, v_encoder_optimizer, v_decoder_optimizer,
              args.n_iters, args.n_pretrain_iters, val_envs=val_envs)  # TODO 先用增强数据集训练

    train(args, train_env, agent, encoder_optimizer, decoder_optimizer, v_encoder_optimizer, v_decoder_optimizer,
          args.n_iters, val_envs=val_envs)  # TODO 再用原始数据集微调


# Test set prediction will be handled separately
# def test_submission(args):
#     ''' Train on combined training and validation sets, and generate test
#     submission. '''
#     agent, train_env, test_envs = test_setup(args)
#     train(args, train_env, agent)
#
#     test_env = test_envs['test']
#     agent.env = test_env
#
#     agent.results_path = '%s%s_%s_iter_%d.json' % (
#         RESULT_DIR, get_model_prefix(args, train_env.image_features_list),
#         'test', args.n_iters)
#     agent.test(use_dropout=False, feedback='argmax')
#     if not args.no_save:
#         agent.write_results()

# python train.py --feedback_method=teacher --use_pretraining --pretrain_splits=dataset
def make_arg_parser():
    parser = argparse.ArgumentParser()
    ImageFeatures.add_args(parser)
    parser.add_argument(
        "--feedback_method", choices=["sample", "teacher", "teacher+sample"],
        default="sample")  # TODO 训练时得是teacher吧？
    parser.add_argument("--bidirectional", action='store_true')  # TODO 有'store_true'的默认值是False
    parser.add_argument("--n_iters", type=int, default=20000)
    parser.add_argument("--use_pretraining", action='store_true')
    parser.add_argument("--pretrain_splits", nargs="+", default=[])  # 决定是否使用数据增强
    parser.add_argument("--n_pretrain_iters", type=int, default=50000)
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument(
        "--use_train_subset", action='store_true',
        help="use a subset of the original train data for validation")
    parser.add_argument("--use_test_set", action='store_true')
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), train_val)
