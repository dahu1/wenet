# Copyright (c) 2023 ASLP@NWPU (authors: Kaixun Huang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


import torch
import torch.nn as nn
from typing import Tuple
from wenet.transformer.attention import MultiHeadedAttention

import logging
logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger('streaming-interface')
logger.setLevel(logging.INFO)
debug = 1
debug = 0


class BLSTM(torch.nn.Module):
    """Context encoder, encoding unequal-length context phrases
       into equal-length embedding representations.
       热词编码器（Context Encoder）：我们使用 BLSTM 作为热词编码器对热词短语进行编码。
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 num_layers,
                 dropout=0.0):
        super(BLSTM, self).__init__()
        self.vocab_size = vocab_size
        # logger.info('vocab_size: {}'.format(vocab_size))
        self.embedding_size = embedding_size
        self.word_embedding = torch.nn.Embedding(
            self.vocab_size, self.embedding_size)

        self.sen_rnn = torch.nn.LSTM(input_size=self.embedding_size,
                                     hidden_size=self.embedding_size,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     batch_first=True,
                                     bidirectional=True)

    def forward(self, sen_batch, sen_lengths):
        sen_batch = torch.clamp(sen_batch, 0)
        # logger.info('sen_batch: {}'.format(sen_batch))
        sen_batch = self.word_embedding(sen_batch)
        pack_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sen_batch, sen_lengths.to('cpu').type(torch.int32),
            batch_first=True, enforce_sorted=False)
        _, last_state = self.sen_rnn(pack_seq)
        laste_h = last_state[0]
        laste_c = last_state[1]
        state = torch.cat([laste_h[-1, :, :], laste_h[-2, :, :],
                          laste_c[-1, :, :], laste_c[-2, :, :]], dim=-1)
        return state


class ContextModule(torch.nn.Module):
    """Context module, Using context information for deep contextual bias

    During the training process, the original parameters of the ASR model
    are frozen, and only the parameters of context module are trained.

    Args:
        vocab_size (int): vocabulary size
        embedding_size (int): number of ASR encoder projection units
        encoder_layers (int): number of context encoder layers
        attention_heads (int): number of heads in the biasing layer
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        encoder_layers: int = 2,
        attention_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.vocab_size = vocab_size
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        # 热词编码器
        self.context_extractor = BLSTM(self.vocab_size, self.embedding_size,
                                       self.encoder_layers)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.embedding_size * 4, self.embedding_size),
            nn.LayerNorm(self.embedding_size)
        )

        # 热词偏置层
        self.biasing_layer = MultiHeadedAttention(
            n_head=self.attention_heads,
            n_feat=self.embedding_size,
            dropout_rate=self.dropout_rate
        )

        # 组合器
        self.combiner = nn.Linear(self.embedding_size, self.embedding_size)
        self.norm_aft_combiner = nn.LayerNorm(self.embedding_size)

        # 热词短语预测网络
        self.context_decoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.context_decoder_ctc_linear = nn.Linear(self.embedding_size,
                                                    self.vocab_size)

        # 热词 CTC 损失
        self.bias_loss = torch.nn.CTCLoss(reduction="sum", zero_infinity=True)

    def forward_context_emb(self,
                            context_list: torch.Tensor,
                            context_lengths: torch.Tensor
                            ) -> torch.Tensor:
        """Extracting context embeddings
           h(ce),热词embding
        """
        # INFO: context_list.shape: torch.Size([2, 3])      
        # INFO: context_lengths.shape: torch.Size([2])
        # INFO: context_emb.shape: torch.Size([2, 2048])
        # INFO: after context_encoder context_emb.shape: torch.Size([1, 2, 512])

        # logger.info('context_list: {} context_list.shape: {}'.format(context_list, context_list.shape))
        # logger.info('context_lengths: {} context_lengths.shape: {}'.format(context_lengths, context_lengths.shape))

        context_emb = self.context_extractor(context_list, context_lengths)
        # logger.info('context_emb.shape: {}'.format(context_emb.shape))

        context_emb = self.context_encoder(context_emb.unsqueeze(0))
        # logger.info('after context_encoder context_emb.shape: {}'.format(context_emb.shape))

        if debug:
            logger.info('context_list: {} context_lengths: {}'.format(context_list, context_lengths))
            logger.info('context_list.shape: {} context_lengths.shape: {}'.format(context_list.shape, context_lengths.shape))
            # context_list.shape: torch.Size([2, 3])
        return context_emb

    def forward(self,
                context_emb: torch.Tensor,
                encoder_out: torch.Tensor,
                biasing_score: float = 1.0,
                recognize: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Using context embeddings for deep biasing.

        Args:
            biasing_score (int): degree of context biasing
            recognize (bool): no context decoder computation if True
        """
        # INFO: encoder_out.shape: torch.Size([1, 329, 512])
        # INFO: context_emb.shape: torch.Size([1, 2, 512])
        # INFO: after expand context_emb.shape: torch.Size([1, 2, 512])
        # INFO: after biasing context_emb.shape: torch.Size([1, 329, 512])
        # INFO: encoder_bias_out.shape: torch.Size([1, 329, 512])

        # logger.info('encoder_out.shape: {}'.format(encoder_out.shape))
        # logger.info('context_emb.shape: {}'.format(context_emb.shape))

        context_emb = context_emb.expand(encoder_out.shape[0], -1, -1)
        # # 这里是对context_emb 做扩展，即接收多个encoder_out，但还是用同一个热词列表
        # # 所以这里可以改进，改成对每个encoder_out 生成一个新的context_emb，需要对热词少的进行补0，以保证结构
        # logger.info('after expand context_emb.shape: {}'.format(context_emb.shape))

        context_emb, _ = self.biasing_layer(encoder_out, context_emb,
                                            context_emb)
        # logger.info('after biasing context_emb.shape: {}'.format(context_emb.shape))

        encoder_bias_out = \
            self.norm_aft_combiner(encoder_out +
                                   self.combiner(context_emb) * biasing_score)
        # logger.info('encoder_bias_out.shape: {}'.format(encoder_bias_out.shape))
        # 推理时 走到 combiner 就输出，作为新的encoder out
        # if recognize:
        #     return encoder_bias_out, torch.tensor(0.0)
        return encoder_bias_out, torch.tensor(0.0)

        # # 训练时还得走 cppn ，作为热词短语网络辅助训练。输出loss_bias
        # bias_out = self.context_decoder(context_emb)
        # bias_out = self.context_decoder_ctc_linear(bias_out)
        # return encoder_bias_out, bias_out
