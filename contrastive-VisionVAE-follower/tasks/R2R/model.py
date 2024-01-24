import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import try_cuda

from env import ConvolutionalImageFeatures, BottomUpImageFeatures


def make_image_attention_layers(args, image_features_list, hidden_size):
    image_attention_size = args.image_attention_size or hidden_size
    attention_mechs = []
    for featurizer in image_features_list:
        if isinstance(featurizer, ConvolutionalImageFeatures):
            if args.image_attention_type == 'feedforward':
                attention_mechs.append(MultiplicativeImageAttention(
                    hidden_size, image_attention_size,
                    image_feature_size=featurizer.feature_dim))
            elif args.image_attention_type == 'multiplicative':
                attention_mechs.append(FeedforwardImageAttention(
                    hidden_size, image_attention_size,
                    image_feature_size=featurizer.feature_dim))
        elif isinstance(featurizer, BottomUpImageFeatures):
            attention_mechs.append(BottomUpImageAttention(
                hidden_size,
                args.bottom_up_detection_embedding_size,
                args.bottom_up_detection_embedding_size,
                image_attention_size,
                featurizer.num_objects,
                featurizer.num_attributes,
                featurizer.feature_dim
            ))
        else:
            attention_mechs.append(None)
    attention_mechs = [
        try_cuda(mech) if mech else mech for mech in attention_mechs]
    return attention_mechs


# TODO: try variational dropout (or zoneout?)
class EncoderLSTM(nn.Module):  # 常规的nlp的encoder
    """ Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. """

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,  # hidden_size = 512
                 dropout_ratio, bidirectional=False, num_layers=1, glove=None):  # bidirectional默认是False
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.use_glove = glove is not None
        if self.use_glove:  # 加载预训练好的embedding
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions)

    def init_state(self, batch_size):
        """ Initialize to zero cell states and hidden states."""
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def forward(self, inputs, lengths):
        """ Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. """
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)  # (batch x seq_len x embedding_size)
        if not self.use_glove:
            embeds = self.drop(embeds)
        h0, c0 = self.init_state(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)  # 循环神经网络常规操作
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))  # encoder最终输出

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)  # 横向连接
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch x hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))  # decoder的初始值

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)  # 恢复矩阵张量形式
        ctx = self.drop(ctx)
        # (batch, seq_len, hidden_size*num_directions), (batch, hidden_size)
        return ctx, decoder_init, c_t


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        """Propagate h through the network.
        dim == hidden_size
        h: (batch x dim)，(batch x hidden_size)
        context: (batch x seq_len x dim)，(batch x seq_len x hidden_size)
        mask: (batch x seq_len) indices to be masked
        """
        target = self.linear_in(h).unsqueeze(2)  # (batch x dim x 1)

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # (batch x seq_len)
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)  # TODO 计算句子中每个单词对预测action贡献的权重
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # (batch x 1 x seq_len)

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim，(batch x hidden_size)
        h_tilde = torch.cat((weighted_context, h), 1)  # (batch x hidden_size+hidden_size)，TODO 将图像和句子的attention连接

        h_tilde = self.tanh(self.linear_out(h_tilde))  # (batch x hidden_size)
        return h_tilde, attn


class ContextOnlySoftDotAttention(nn.Module):
    """Like SoftDot, but don't concatenat h or perform the non-linearity transform
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim, context_dim=None):
        """Initialize layer."""
        super(ContextOnlySoftDotAttention, self).__init__()
        if context_dim is None:
            context_dim = dim
        self.linear_in = nn.Linear(dim, context_dim, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        """Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weighted_context, attn


class FeedforwardImageAttention(nn.Module):
    def __init__(self, context_size, hidden_size, image_feature_size=2048):
        super(FeedforwardImageAttention, self).__init__()
        self.feature_size = image_feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.fc1_feature = nn.Conv2d(
            image_feature_size, hidden_size, kernel_size=1, bias=False)
        self.fc1_context = nn.Linear(context_size, hidden_size, bias=True)
        self.fc2 = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        feature_hidden = self.fc1_feature(feature)
        context_hidden = self.fc1_context(context)
        context_hidden = context_hidden.unsqueeze(-1).unsqueeze(-1)
        x = feature_hidden + context_hidden
        x = self.fc2(F.relu(x))
        # batch_size x (width * height) x 1
        attention = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1)
        # batch_size x feature_size x (width * height)
        reshaped_features = feature.view(batch_size, self.feature_size, -1)
        x = torch.bmm(reshaped_features, attention)  # batch_size x
        return x.squeeze(-1), attention.squeeze(-1)


class MultiplicativeImageAttention(nn.Module):
    def __init__(self, context_size, hidden_size, image_feature_size=2048):
        super(MultiplicativeImageAttention, self).__init__()
        self.feature_size = image_feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.fc1_feature = nn.Conv2d(
            image_feature_size, hidden_size, kernel_size=1, bias=True)
        self.fc1_context = nn.Linear(context_size, hidden_size, bias=True)
        self.fc2 = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        # batch_size x hidden_size x width x height
        feature_hidden = self.fc1_feature(feature)
        # batch_size x hidden_size
        context_hidden = self.fc1_context(context)
        # batch_size x 1 x hidden_size
        context_hidden = context_hidden.unsqueeze(-2)
        # batch_size x hidden_size x (width * height)
        feature_hidden = feature_hidden.view(batch_size, self.hidden_size, -1)
        # batch_size x 1 x (width x height)
        x = torch.bmm(context_hidden, feature_hidden)
        # batch_size x (width * height) x 1
        attention = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1)
        # batch_size x feature_size x (width * height)
        reshaped_features = feature.view(batch_size, self.feature_size, -1)
        x = torch.bmm(reshaped_features, attention)  # batch_size x
        return x.squeeze(-1), attention.squeeze(-1)


class BottomUpImageAttention(nn.Module):
    def __init__(self, context_size, object_embedding_size,
                 attribute_embedding_size, hidden_size, num_objects,
                 num_attributes, image_feature_size=2048):
        super(BottomUpImageAttention, self).__init__()
        self.context_size = context_size
        self.object_embedding_size = object_embedding_size
        self.attribute_embedding_size = attribute_embedding_size
        self.hidden_size = hidden_size
        self.num_objects = num_objects
        self.num_attributes = num_attributes
        self.feature_size = (image_feature_size + object_embedding_size +
                             attribute_embedding_size + 1 + 5)

        self.object_embedding = nn.Embedding(
            num_objects, object_embedding_size)
        self.attribute_embedding = nn.Embedding(
            num_attributes, attribute_embedding_size)

        self.fc1_context = nn.Linear(context_size, hidden_size)
        self.fc1_feature = nn.Linear(self.feature_size, hidden_size)
        # self.fc1 = nn.Linear(context_size + self.feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, bottom_up_features, context):
        # image_features: batch_size x max_num_detections x feature_size
        # object_ids: batch_size x max_num_detections
        # attribute_ids: batch_size x max_num_detections
        # no_object_mask: batch_size x max_num_detections
        # context: batch_size x context_size

        # batch_size x max_num_detections x embedding_size
        attribute_embedding = self.attribute_embedding(
            bottom_up_features.attribute_indices)
        # batch_size x max_num_detections x embedding_size
        object_embedding = self.object_embedding(
            bottom_up_features.object_indices)
        # batch_size x max_num_detections x (feat size)
        feats = torch.cat((
            bottom_up_features.cls_prob.unsqueeze(2),
            bottom_up_features.image_features,
            attribute_embedding, object_embedding,
            bottom_up_features.spatial_features), dim=2)

        # attended_feats = feats.mean(dim=1)
        # attention = None

        # batch_size x 1 x hidden_size
        x_context = self.fc1_context(context).unsqueeze(1)
        # batch_size x max_num_detections x hidden_size
        x_feature = self.fc1_feature(feats)
        # batch_size x max_num_detections x hidden_size
        x = x_context * x_feature
        x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        x = self.fc2(x).squeeze(-1)  # batch_size x max_num_detections
        x.data.masked_fill_(bottom_up_features.no_object_mask, -float("inf"))
        # batch_size x 1 x max_num_detections
        attention = F.softmax(x, 1).unsqueeze(1)
        # batch_size x feat_size
        attended_feats = torch.bmm(attention, feats).squeeze(1)
        return attended_feats, attention


class VisualSoftDotAttention(nn.Module):
    """ Visual Dot Attention Layer. """

    def __init__(self, h_dim, v_dim, dot_dim=256):
        """Initialize layer."""
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)
        # self.linear_out_w = nn.Linear(v_dim, v_dim, bias=True)  # 将图像向量范围投影到[-1,1]，与重建图像向量范围一致
        # self.th = nn.Tanh()

    def forward(self, h, visual_context, mask=None):
        """Propagate h through the network.

        h: (batch x h_dim)
        visual_context: (batch x v_num x v_dim)
        """
        target = self.linear_in_h(h).unsqueeze(2)  # (batch x dot_dim x 1)
        context = self.linear_in_v(visual_context)  # (batch x v_num x dot_dim)

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # (batch x v_num)，得到36个视点的attention
        attn = self.sm(attn)  # TODO 计算出36个全景图像对预测action的贡献权重
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # (batch x 1 x v_num)

        # TODO 计算全景图像的attention
        weighted_context = torch.bmm(
            attn3, visual_context).squeeze(1)  # (batch x v_dim),TODO 数值可能小于0也可能大于0
        # weighted_context = self.linear_out_w(weighted_context)
        # weighted_context = self.th(weighted_context)
        return weighted_context, attn


# TODO 预测action
class EltwiseProdScoring(nn.Module):
    """
    Linearly mapping h and v to the same dimension, and do a elementwise
    multiplication and a linear scoring
    """

    def __init__(self, h_dim, a_dim, dot_dim=256):
        """Initialize layer."""
        super(EltwiseProdScoring, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        """Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        """
        target = self.linear_in_h(h).unsqueeze(1)  # batch x 1 x dot_dim
        context = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, context)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits


class VisionAttnEecoderLSTM(nn.Module):
    """
    An unrolled LSTM with attention over visual_context for encoding panoramic image information
    """

    def __init__(self, hidden_size, dropout_ratio,  # hidden_size = 512
                 feature_size=2048 + 128, z_dim=32, image_attention_layers=None):
        super(VisionAttnEecoderLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(feature_size, hidden_size)

        self.visual_attention_layer = VisualSoftDotAttention(
            hidden_size, feature_size)

        # VAE模块
        # VAE encoder
        self.z_dim = z_dim
        self.linear_mu = nn.Linear(hidden_size, z_dim)  # mu
        self.linear_logsigma = nn.Linear(hidden_size, z_dim)  # log_sigma

    def init_state(self, batch_size):
        """ Initialize to zero cell states and hidden states."""
        h0 = Variable(torch.zeros(
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def VAE_encode(self, h_t):
        mu = self.linear_mu(h_t)
        log_sigma = self.linear_logsigma(h_t)  # estimate log(sigma**2) actually
        return mu, log_sigma

    def VAE_reparameterzie(self, mu, log_sigma):
        std = torch.exp(log_sigma * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, visual_context, h_0, c_0):
        """ Takes a single step in the decoder LSTM (allowing sampling).

        visual_context: (batch x v_num x feature_size)，v_num = 36
        h_0: (batch x hidden_size)，decoder上一时刻的输出
        c_0: (batch x hidden_size)
        """
        feature, alpha_v = self.visual_attention_layer(h_0, visual_context)  # feature：(batch x 2048+128)
        drop = self.drop(feature)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))  # TODO (batch x hidden_size)，将h_1用于对比学习
        h_1_drop = self.drop(h_1)

        # VAE处理
        mu, log_sigma = self.VAE_encode(h_1_drop)
        sampled_z = self.VAE_reparameterzie(mu, log_sigma)  # TODO VAE encode
        # mu, log_sigma, sampled_z: (batch x z_dim)
        return feature, h_1, h_1_drop, c_1, alpha_v, sampled_z, mu, log_sigma


class VisionDecoderLSTM(nn.Module):
    """
    An unrolled LSTM for reconstructing image vector sequence
    actions.
    """

    def __init__(self, hidden_size, dropout_ratio,  # hidden_size = 512
                 feature_size=2048 + 128, z_dim=32, image_attention_layers=None):
        super(VisionDecoderLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(hidden_size, feature_size)

        # VAE模块
        # VAE decoder
        self.z_dim = z_dim
        self.linear_z2h = nn.Linear(z_dim, hidden_size)
        self.linear_h2h = nn.Linear(hidden_size, hidden_size)
        self.th = nn.Tanh()

        self.linear_out = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.Linear(feature_size, feature_size),
        )

    def init_state(self, batch_size):
        """ Initialize to zero cell states and hidden states."""
        h0 = Variable(torch.zeros(
            batch_size,
            self.feature_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            batch_size,
            self.feature_size
        ), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def VAE_decode(self, z):
        h_t = self.linear_z2h(z)
        h_t = self.linear_h2h(h_t)
        h_t = self.th(h_t)
        return h_t

    # def forward(self, ve_h_0, h_0, c_0):
    #     """ Takes a single step in the decoder LSTM (allowing sampling).
    #
    #     ve_h_0: (batch x hidden_size)
    #     h_0: (batch x feature_size)，decoder上一时刻的输出
    #     c_0: (batch x feature_size)
    #     """
    #     drop = self.drop(ve_h_0)  # (batch x hidden_size)
    #     h_1, c_1 = self.lstm(drop, (h_0, c_0))  # (batch x feature_size)
    #     h_1_drop = self.drop(h_1)
    #     re_feature = self.linear_out(h_1_drop)
    #     return re_feature, h_1, c_1

    def forward(self, z, h_0, c_0):
        """ Takes a single step in the decoder LSTM (allowing sampling).

        z: (batch x z_dim)，VAE encoder的输出
        h_0: (batch x feature_size)，decoder上一时刻的输出
        c_0: (batch x feature_size)
        """
        ve_h_0 = self.VAE_decode(z)  # TODO VAE decode
        drop = self.drop(ve_h_0)  # (batch x hidden_size)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))  # (batch x feature_size)
        h_1_drop = self.drop(h_1)
        re_feature = self.linear_out(h_1_drop)
        return re_feature, h_1, c_1


class LangAttnDecoderLSTM(nn.Module):
    """
    An unrolled LSTM with attention over instructions for decoding navigation
    actions.
    """

    def __init__(self, embedding_size, hidden_size, dropout_ratio,  # hidden_size = 512
                 feature_size=2048 + 128, image_attention_layers=None):
        super(LangAttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.u_begin = try_cuda(Variable(
            torch.zeros(embedding_size), requires_grad=False))  # TODO action：(2048+128)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)

        self.text_attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = EltwiseProdScoring(hidden_size, embedding_size)

    def forward(self, u_t_prev, all_u_t, vd_feature_t, h_0, c_0, ctx,
                ctx_mask=None):
        """ Takes a single step in the decoder LSTM (allowing sampling).

        u_t_prev: (batch x embedding_size)
        all_u_t: (batch x a_num x embedding_size)
        vd_feature_t: (batch x  feature_size)，v_decoder的输出
        h_0: (batch x hidden_size)，decoder上一时刻的输出
        c_0: (batch x hidden_size)
        ctx: (batch x seq_len x dim)
        ctx_mask: (batch x seq_len) - indices to be masked
        """
        # 将action信息和重建图像attention连接起来
        concat_input = torch.cat((u_t_prev, vd_feature_t), 1)  # (batch x 2048+128) + (batch x 2048+128)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))  # (batch x hidden_size)
        h_1_drop = self.drop(h_1)
        # 句子的ctx和(动作+图像)embedding做attention
        h_tilde, alpha = self.text_attention_layer(h_1_drop, ctx, ctx_mask)  # h_tilde：(batch x hidden_size)
        logit = self.decoder2action(h_tilde, all_u_t)  # (batch x a_num)，得到对每个action的预测值
        return h_1, c_1, alpha, logit


###############################################################################
# speaker models
###############################################################################


class SpeakerEncoderLSTM(nn.Module):
    def __init__(self, action_embedding_size, world_embedding_size,
                 hidden_size, dropout_ratio, bidirectional=False):
        super(SpeakerEncoderLSTM, self).__init__()
        assert not bidirectional, 'Bidirectional is not implemented yet'

        self.action_embedding_size = action_embedding_size
        self.word_embedding_size = world_embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.visual_attention_layer = VisualSoftDotAttention(
            hidden_size, world_embedding_size)
        self.lstm = nn.LSTMCell(
            action_embedding_size + world_embedding_size, hidden_size)
        self.encoder2decoder = nn.Linear(hidden_size, hidden_size)

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(
            torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        c0 = Variable(
            torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def _forward_one_step(self, h_0, c_0, action_embedding,
                          world_state_embedding):
        feature, _ = self.visual_attention_layer(h_0, world_state_embedding)
        concat_input = torch.cat((action_embedding, feature), 1)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        return h_1, c_1

    def forward(self, batched_action_embeddings, world_state_embeddings):
        ''' Expects action indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        assert isinstance(batched_action_embeddings, list)
        assert isinstance(world_state_embeddings, list)
        assert len(batched_action_embeddings) == len(world_state_embeddings)
        batch_size = world_state_embeddings[0].shape[0]

        h, c = self.init_state(batch_size)
        h_list = []
        for t, (action_embedding, world_state_embedding) in enumerate(
                zip(batched_action_embeddings, world_state_embeddings)):
            h, c = self._forward_one_step(
                h, c, action_embedding, world_state_embedding)
            h_list.append(h)

        decoder_init = nn.Tanh()(self.encoder2decoder(h))

        ctx = torch.stack(h_list, dim=1)  # (batch, seq_len, hidden_size)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c  # (batch, hidden_size)


class SpeakerDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, vocab_embedding_size, hidden_size,
                 dropout_ratio, glove=None, use_input_att_feed=False):
        super(SpeakerDecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.drop = nn.Dropout(p=dropout_ratio)
        self.use_input_att_feed = use_input_att_feed
        if self.use_input_att_feed:
            print('using input attention feed in SpeakerDecoderLSTM')
            self.lstm = nn.LSTMCell(
                vocab_embedding_size + hidden_size, hidden_size)
            self.attention_layer = ContextOnlySoftDotAttention(hidden_size)
            self.output_l1 = nn.Linear(hidden_size * 2, hidden_size)
            self.tanh = nn.Tanh()
        else:
            self.lstm = nn.LSTMCell(vocab_embedding_size, hidden_size)
            self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, vocab_size)

    def forward(self, previous_word, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        word_embeds = self.embedding(previous_word)
        word_embeds = word_embeds.squeeze()  # (batch, embedding_size)
        if not self.use_glove:
            word_embeds_drop = self.drop(word_embeds)
        else:
            word_embeds_drop = word_embeds

        if self.use_input_att_feed:
            h_tilde, alpha = self.attention_layer(
                self.drop(h_0), ctx, ctx_mask)
            concat_input = torch.cat((word_embeds_drop, self.drop(h_tilde)), 1)
            h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
            x = torch.cat((h_1, h_tilde), 1)
            x = self.drop(x)
            x = self.output_l1(x)
            x = self.tanh(x)
            logit = self.decoder2action(x)
        else:
            h_1, c_1 = self.lstm(word_embeds_drop, (h_0, c_0))
            h_1_drop = self.drop(h_1)
            h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
            logit = self.decoder2action(h_tilde)
        return h_1, c_1, alpha, logit
