""" Utils for io, language, connectivity graphs etc """

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
import subprocess
import itertools
import base64

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']

vocab_pad_idx = base_vocab.index('<PAD>')  # 0
vocab_unk_idx = base_vocab.index('<UNK>')  # 1
vocab_eos_idx = base_vocab.index('<EOS>')  # 2
vocab_bos_idx = base_vocab.index('<BOS>')  # TODO 这个从来没用过，一般decoder解码句子的时候才需要


# TODO 加载每个场景的加权无向图
def load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


# TODO 加载R2R数据集
def load_datasets(splits):
    data = []
    for split in splits:
        with open('tasks/R2R/data/R2R_%s.json' % split) as f:
            data += json.load(f)
        # with open('data/R2R_%s.json' % split) as f:  # TODO 路径可能要这样写
        #     data += json.load(f)

    # {"distance": 12.59, "scan": "VFuaQ6m2Qom", "path_id": 1981,
    # "path": ["5d50911d1c074a3c8de82d35bc4c558b", "8ba5f3cf31934fc98cde3cf4a1116551", "6c8329a5bbd5423696d5d9a22237d0f8", "1ed9136647664140918246b69b5d2dc5", "9377f3ca210946ff9dbea4937cf7d3ad", "8c31225bb638494082b206e492422ebf", "2185b2e2cb704157aefd1dd81f5f3811"],
    # "heading": 3.803,
    # "instructions": ["Walk straight through to doorway on the other side of the room. Turn left and stop on the stairs near the table. ", "Walk straight past the hot tubs.  Go through the doorway and turn to the right.  Then veer to the left and stop there.  Wait. ", "walk forward with the pool on your left. Enter the house and take a right. Go forward into the living room and stop in the living room doorway. "]}
    return data


# TODO 避免字符串乱码
def decode_base64(string):
    if sys.version_info[0] == 2:
        return base64.decodestring(bytearray(string))
    elif sys.version_info[0] == 3:
        return base64.decodebytes(bytearray(string, 'utf-8'))
    else:
        raise ValueError("decode_base64 can't handle python version {}".format(sys.version_info[0]))


class Tokenizer(object):
    """ Class to tokenize and encode a sentence. """
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, vocab=None):
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

    @staticmethod
    def split_sentence(sentence):
        """ Break sentence into a list of words and punctuation """
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in Tokenizer.split_sentence(sentence):
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(vocab_unk_idx)
        # encoding.append(vocab_eos_idx)
        # utterance_length = len(encoding)
        # if utterance_length < self.encoding_length:
        # encoding += [vocab_pad_idx] * (self.encoding_length - len(encoding))
        # encoding = encoding[:self.encoding_length] # leave room for unks
        arr = np.array(encoding)  # TODO 没加<EOS>和<PAD>！
        return arr, len(encoding)

    def decode_sentence(self, encoding, break_on_eos=False, join=True):
        sentence = []
        for ix in encoding:
            if ix == (vocab_eos_idx if break_on_eos else vocab_pad_idx):
                break
            else:
                sentence.append(self.vocab[ix])
        if join:
            return " ".join(sentence)  # 返回字符串
        return sentence  # 返回列表


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    """ Build a vocab, starting with base vocab containing a few useful tokens. """
    count = Counter()
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(Tokenizer.split_sentence(instr))
    vocab = list(start_vocab)
    for word, num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab), path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)  # 一行一个单词


def read_vocab(path):
    vocab = []
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since  # 当前已花费时间
    es = s / percent  # 预计要花费的总时间
    rs = es - s  # 预计剩下的时间
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def k_best_indices(arr, k, sorted=False):
    # https://stackoverflow.com/a/23734295
    if k >= len(arr):
        if sorted:
            return np.argsort(arr)
        else:
            return np.arange(0, len(arr))
    ind = np.argpartition(arr, -k)[-k:]
    if sorted:
        ind = ind[np.argsort(arr[ind])]
    return ind


# TODO 不懂
def structured_map(function, *args, **kwargs):
    # assert all(len(a) == len(args[0]) for a in args[1:])
    nested = kwargs.get('nested', False)
    acc = []
    for t in zip(*args):
        if nested:
            mapped = [function(*inner_t) for inner_t in zip(*t)]
        else:
            mapped = function(*t)
        acc.append(mapped)
    return acc


def flatten(lol):
    return [l for lst in lol for l in lst]


def all_equal(lst):
    return all(x == lst[0] for x in lst[1:])


# TODO 把数据放到cuda上
def try_cuda(pytorch_obj):
    import torch.cuda
    try:
        disabled = torch.cuda.disabled
    except:
        disabled = False
    if torch.cuda.is_available() and not disabled:
        return pytorch_obj.cuda()
    else:
        return pytorch_obj


def pretty_json_dump(obj, fp):
    json.dump(obj, fp, sort_keys=True, indent=4, separators=(',', ':'))


def spatial_feature_from_bbox(bboxes, im_h, im_w):
    # from Ronghang Hu
    # https://github.com/ronghanghu/cmn/blob/ff7d519b808f4b7619b17f92eceb17e53c11d338/models/spatial_feat.py

    # Generate 5-dimensional spatial features from the image
    # [xmin, ymin, xmax, ymax, S] where S is the area of the box
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))
    # Check the size of the bounding boxes
    assert np.all(bboxes[:, 0:2] >= 0)
    assert np.all(bboxes[:, 0] <= bboxes[:, 2])
    assert np.all(bboxes[:, 1] <= bboxes[:, 3])
    assert np.all(bboxes[:, 2] <= im_w)
    assert np.all(bboxes[:, 3] <= im_h)

    feats = np.zeros((bboxes.shape[0], 5), dtype=np.float32)
    feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
    feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
    feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
    feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
    feats[:, 4] = (feats[:, 2] - feats[:, 0]) * (feats[:, 3] - feats[:, 1])  # S
    return feats


# TODO 执行程序的入口
def run(arg_parser, entry_function):
    arg_parser.add_argument("--pdb", action='store_true')
    arg_parser.add_argument("--ipdb", action='store_true')
    arg_parser.add_argument("--no_cuda", action='store_true')

    args = arg_parser.parse_args()

    import torch.cuda
    # todo: yuck
    torch.cuda.disabled = args.no_cuda

    def log(out_file):
        subprocess.call("git rev-parse HEAD", shell=True, stdout=out_file)
        subprocess.call("git --no-pager diff", shell=True, stdout=out_file)
        out_file.write('\n\n')
        out_file.write(' '.join(sys.argv))
        out_file.write('\n\n')
        json.dump(vars(args), out_file)
        out_file.write('\n\n')

    log(sys.stdout)
    # if 'save_dir' in vars(args) and args.save_dir:
    #     with open(os.path.join(args.save_dir, 'invoke.log'), 'w') as f:
    #         log(f)

    if args.ipdb:
        import ipdb
        ipdb.runcall(entry_function, args)
    elif args.pdb:
        import pdb
        pdb.runcall(entry_function, args)
    else:
        entry_function(args)
