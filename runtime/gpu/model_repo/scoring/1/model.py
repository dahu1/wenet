# Copyright (c) 2021 NVIDIA CORPORATION
#               2023 58.com(Wuba) Inc AI Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton_python_backend_utils as pb_utils
import numpy as np
import multiprocessing
from torch.utils.dlpack import from_dlpack
import torch
from swig_decoders import ctc_beam_search_decoder_batch, \
    Scorer, HotWordsScorer, PathTrie, TrieVector, map_batch
import json
import os
import yaml
import re
from utils import log_add, ctc_prefix_beam_search_batch, ContextGraph, read_symbol_table, tokenize_dict, tokenize_list

p = re.compile(r'▁')

import copy
import math
import onnxruntime
import logging
import torch.nn.functional as F
from context_module import  LocalModel
logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger('streaming-interface')
logger.setLevel(logging.INFO)

debug = 0
# debug = 1
SeeDeepBiasWordDebug = 1

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")
        self.out1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])

        output2_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT2")
        self.out2_dtype = pb_utils.triton_string_to_numpy(output2_config['data_type'])

        # Get INPUT configuration
        batch_log_probs = pb_utils.get_input_config_by_name(
            model_config, "batch_log_probs")
        self.beam_size = batch_log_probs['dims'][-1]

        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        self.feature_size = encoder_config['dims'][-1]

        self.lm = None
        self.hotwords_scorer = None
        self.init_ctc_rescore(self.model_config['parameters'])
        print('Initialized Rescoring!')


    def init_ctc_rescore(self, parameters):
        bidecoder = 0
        lm_path, vocab_path = None, None
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "lm_path":
                lm_path = value
            elif key == "hotwords_path":
                hotwords_path = value
            elif key == "vocabulary":
                vocab_path = value
            elif key == "bidecoder":
                bidecoder = int(value)
            elif key == "deep_biasing_score":
                deep_biasing_score = float(value)
            elif key == "graph_biasing":
                graph_biasing = bool(int(value))
            elif key == "deep_biasing":
                deep_biasing = bool(int(value))
            elif key == "context_pt":
                context_pt = value

        _, vocab = self.load_vocab(vocab_path)
        self.vocab_path = vocab_path
        self.vocab = vocab

        self.vocabulary = vocab
        self.bidecoder = bidecoder
        sos = eos = len(vocab) - 1
        self.sos = sos
        self.eos = eos

        # 加载预设的热词
        # graph context 
        self.symbol_table = read_symbol_table(self.vocab_path)

        self.context_graph = ContextGraph()
        self.context_graph.load_context_path(hotwords_path)
        self.context_graph.context_tonkenize(self.symbol_table)

        self.context_graph.graph_biasing = graph_biasing
        self.context_graph.deep_biasing = deep_biasing
        self.context_graph.deep_biasing_score = deep_biasing_score
        self.context_graph.build_graph_dict(self.context_graph.context_list1)

        # deep bias context model
        checkpoint = torch.load(context_pt)
        # logger.info('self.vocab: {} len: {}'.format(self.vocab, len(self.vocab)))

        localmodel  = LocalModel(len(self.vocab))
        localmodel.load_state_dict(checkpoint, strict=False)
        localmodel.state_dict()["context_module.context_decoder_ctc_linear.weight"] = checkpoint["ctc.ctc_lo.weight"]
        localmodel.state_dict()["context_module.context_decoder_ctc_linear.bias"] = checkpoint["ctc.ctc_lo.bias"]

        use_cuda = True
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.context_module = localmodel.context_module.to(self.device)
        self.context_module.eval()
        # logger.info('self.context_pt: {}'.format(self.context_module))
        logger.info('self.context_pt load succeed!')

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        id2vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                char, id = line.split()
                id2vocab[int(id)] = char
        vocab = [0] * len(id2vocab)
        for id, char in id2vocab.items():
            vocab[id] = char
        return id2vocab, vocab

    def load_hotwords(self, hotwords_file):
        """
        load hotwords.yaml
        """
        # with open(hotwords_file, 'r', encoding="utf-8") as fin:
        #     configs = yaml.load(fin, Loader=yaml.FullLoader)
        # return configs
        hotwords_dict = dict()
        for line in  open(hotwords_file, 'r', encoding="utf-8"):
            lin = line.strip().split("\t")
            if len(lin) <= 1 :
                continue
            k = lin[0]
            v = int(lin[1])
            hotwords_dict[k] = v
        return hotwords_dict


    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        batch_encoder_out, batch_encoder_lens = [], []
        batch_count = []

        encoder_max_len = 0
        ctc_log_probs_max_len = 0
        hyps_max_len = 0
        total = 0
        score_hyps = []
        batch_ctc_log_probs = []
        batch_hot_word = []
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            in_1 = pb_utils.get_input_tensor_by_name(request, "encoder_out_lens")
            in_4 = pb_utils.get_input_tensor_by_name(request, "ctc_log_probs")
            in_5 = pb_utils.get_input_tensor_by_name(request, "hot_word")

            batch_encoder_out.append(in_0.as_numpy())
            encoder_max_len = max(encoder_max_len, batch_encoder_out[-1].shape[1])

            batch_ctc_log_probs.append(in_4.as_numpy())
            ctc_log_probs_max_len = max(ctc_log_probs_max_len, batch_ctc_log_probs[-1].shape[1])
            batch_hot_word.append(in_5.as_numpy())

            cur_b_lens = in_1.as_numpy()
            batch_encoder_lens.append(cur_b_lens)
            cur_batch = cur_b_lens.shape[0]
            batch_count.append(cur_batch)

            encoder_out = in_0.as_numpy()
            ctc_log_probs = in_4.as_numpy()
            # batch_ctc_log_probs.append(ctc_log_probs)

            if debug:
                logger.info('get encoder out ')
                logger.info('in_0: {} '.format(in_0))
                logger.info('encoder_out: {} shape: {}'.format(encoder_out, encoder_out.shape))
                logger.info('ctc_log_probs: {} shape: {}'.format(ctc_log_probs, ctc_log_probs.shape))
                logger.info('encoder_out_lens: {} shape: {}'.format(cur_b_lens, cur_b_lens.shape))
                logger.info('in_5: {} as_numpy: {} shape: {}'.format(in_5, in_5.as_numpy(), in_5.as_numpy().shape))
                get_hot_word_dict = str(in_5.as_numpy()[0], encoding = "utf-8")
                logger.info('get_hot_word_dict: {} type: {}'.format(get_hot_word_dict, type(get_hot_word_dict)))
                get_hot_word_dict = json.loads(get_hot_word_dict)
                logger.info('get_hot_word_dict after json: {} type: {}'.format(get_hot_word_dict, type(get_hot_word_dict)))

            for i in range(cur_batch):
                cur_len = cur_b_lens[i]
                total += 1

        logger.info('encoder end')
        # step 合并batch encoder
        feature_size = self.feature_size
        in_encoder_out = np.zeros((total, encoder_max_len, feature_size), dtype=self.data_type)
        in_ctc_log_probs = np.zeros((total, ctc_log_probs_max_len, batch_ctc_log_probs[-1].shape[-1]), dtype=self.data_type)
        in_hot_word = []
        st = 0
        idex = 0
        # 这里考虑到多个请求，每个请求有b个batch，把这些合成一个大的tensor
        # total 即为所有的请求加起来的batch数
        for b in batch_count:
            t = batch_encoder_out[idex]
            t2 = batch_ctc_log_probs[idex]
            in_encoder_out[st:st + b, 0:t.shape[1]] = t      
            in_ctc_log_probs[st:st + b, 0:t2.shape[1]] = t2
            in_hot_word.extend(batch_hot_word[idex])
            st += b
            idex += 1
        if debug:
            logger.info('batch encoder_out ')
            logger.info('in_encoder_out.shape: {}'.format(in_encoder_out.shape))
            logger.info('in_ctc_log_probs.shape: {}'.format(in_ctc_log_probs.shape))

        logger.info('batch merge end')
        # 对送进来的total数量的热词 进行处理，转成graph
        online_graph_list = []
        for b in range(total):
            get_hot_word_dict = str(in_hot_word[b], encoding = "utf-8")
            get_hot_word_dict = json.loads(get_hot_word_dict)
            if len(get_hot_word_dict) == 0:
                online_graph = self.context_graph
            else:
                online_graph = ContextGraph()
                online_graph.graph_biasing = self.context_graph.graph_biasing
                online_graph.deep_biasing = self.context_graph.deep_biasing
                online_graph.context_dict = copy.deepcopy(self.context_graph.context_dict)
                online_graph.context_dict.update(get_hot_word_dict)
                online_graph.hot_word_dict = get_hot_word_dict
                online_graph.context_tonkenize(self.symbol_table)
                online_graph.build_graph_dict(online_graph.context_list1)

            online_graph_list.append(online_graph)
        logger.info('construct online context graph end')

        # 参考in_encoder_out的实现，把ctc_probs 和 context_list搞成类似的
        # in_encoder_out     B x T x F
        # in_ctc_log_probs   B x T x V

        # ==> step deep bias start
        # 这里因为context forward部分，context_emb实现的时候，不支持batch，故单个推理
        if self.context_graph.deep_biasing:
          # logger.info('total: {}'.format(total))
          for b in range(total):
              ctc_probs = torch.from_numpy(in_ctc_log_probs[b]) # T x V
              # encoder_out = in_encoder_out[b].unsqueeze(0)  # 1 * T x V
              encoder_out = np.expand_dims(in_encoder_out[b], axis = 0)  # 1 * T x V
              encoder_out = torch.from_numpy(encoder_out)
              online_graph = online_graph_list[b]
              filtered_context_list = online_graph.two_stage_filtering(online_graph.context_list, ctc_probs)
              # filtered_context_list = online_graph.context_list
              online_graph.filter_hot_word_dict = dict(zip(["".join([self.vocab[i] for i in j]) for j in filtered_context_list], [1 for j in filtered_context_list]))
              # deep biasing 过滤词的策略： 先用两阶段算法过滤一遍所有的热词，
              # 1 先用两阶段算法过滤一遍所有的热词，
              # 2 再额外添加不超过10个传入的词,后续可以改成添加10个权重最高的热词
              idx = 0
              for hot_word in online_graph.hot_word_dict.keys():
                  if hot_word in online_graph.filter_hot_word_dict:
                      continue
                  else:
                      online_graph.filter_hot_word_dict[hot_word] = 1
                      filtered_context_list.extend(tokenize_list([hot_word], self.symbol_table))
                      idx = idx + 1
                  if idx > 10 : 
                      break
              # if len(online_graph.hot_word_dict) <=5 :
              #     filtered_context_list.extend(tokenize_list(online_graph.hot_word_dict.keys(), self.symbol_table))
              if debug:
                  logger.info('see online_graph.filter_hot_word_dict: {} '.format(online_graph.filter_hot_word_dict))
                  logger.info('see filtered after extend  word: {} '.format(["".join([self.vocab[i] for i in j]) for j in filtered_context_list]))
                  logger.info('filtered_context_list: {} '.format(filtered_context_list))
                  logger.info('b: {}'.format(b))
                  logger.info('ctc_probs shape: {}'.format(ctc_probs.shape))
              if SeeDeepBiasWordDebug:
                  # logger.info('filtered_context_list: {} '.format(filtered_context_list))
                  logger.info('see filtered deep bias word: {} '.format(["".join([self.vocab[i] for i in j]) for j in filtered_context_list]))
              # context_list, context_list_lengths = self.context_graph.get_context_list_tensor(filtered_context_list)
              context_list, context_list_lengths = online_graph.get_context_list_tensor(filtered_context_list)

              context_list = context_list.to(self.device)
              context_list_lengths = context_list_lengths.to(self.device)
              encoder_out = encoder_out.to(self.device)

              context_emb = self.context_module.forward_context_emb(context_list, context_list_lengths)
              if debug:
                  logger.info('context_list: {} shape: {} type: {}'.format(context_list, context_list.shape, type(context_list)))
                  logger.info('context_list_lengths: {} shape:{} type: {}'.format(context_list_lengths, context_list_lengths.shape ,type(context_list_lengths)))
                  logger.info('encoder_out shape: {} type: {}'.format(encoder_out.shape, type(encoder_out)))
                  logger.info('context_emb: {}, shape: {}'.format(context_emb, context_emb.shape))
              encoder_out, _ = self.context_module(context_emb, encoder_out, self.context_graph.deep_biasing_score, True)

              in_encoder_out[b] = encoder_out.cpu().detach().numpy()
              # ctc_probs = self.ctc.log_softmax(encoder_out) 因为encoder out变了，所以ctc_probs需要重新计算，原始ctc_probs是encoder.onnx 直接给的，这里手动计算一下。
              ctc_log_prob = self.context_module.context_decoder_ctc_linear(encoder_out)
              in_ctc_log_probs[b] = F.log_softmax(ctc_log_prob , dim=2).cpu().detach().numpy()

        # ==> deep bias end
        logger.info('deep bias forward end')
        # step ctc batch decode 
        # todo: 这里ctc batch decode实现的有问题，需要把ctc log prob放到一个大的batch数组里，遍历计算。 done，已修复。

        score_hyps = ctc_prefix_beam_search_batch(in_encoder_out, self.beam_size, in_ctc_log_probs, online_graph_list)
        logger.info('ctc prefix beam search end')
        for b in range(total):
            online_graph = online_graph_list.pop()

        logger.info('ctc batch : {} '.format(len(score_hyps)))
        if debug:
            logger.info('in_encoder_out :{} shape: {}'.format(in_encoder_out, in_encoder_out.shape))
            logger.info('in_ctc_log_probs :{} shape: {}'.format(in_ctc_log_probs, in_ctc_log_probs.shape))
            logger.info('ctc_prefix_beam_search_batch inference ')
            logger.info('hyps: {} '.format(score_hyps))

        all_hyps = []
        all_ctc_score = []
        all_time_stamp = []
        max_seq_len = 0
        for seq_cand in score_hyps: 
            # if candidates less than beam size
            if len(seq_cand) != self.beam_size:
                seq_cand = list(seq_cand)
                seq_cand += (self.beam_size - len(seq_cand)) * [(-float("INF"), (0,), [])]

            for score, hyps, time_stamp  in seq_cand:
                all_hyps.append(list(hyps))
                all_ctc_score.append(score)
                all_time_stamp.append(time_stamp)
                max_seq_len = max(len(hyps), max_seq_len)

        beam_size = self.beam_size
        hyps_max_len = max_seq_len + 2
        in_ctc_score = np.zeros((total, beam_size), dtype=self.data_type)
        in_hyps_pad_sos_eos = np.ones(
            (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos
        if self.bidecoder:
            in_r_hyps_pad_sos_eos = np.ones(
                (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos

        in_hyps_lens_sos = np.ones((total, beam_size), dtype=np.int32)

        # in_encoder_out = np.zeros((total, encoder_max_len, feature_size),
        #                           dtype=self.data_type)
        in_encoder_out_lens = np.zeros(total, dtype=np.int32)
        in_all_time_stamp = []
        st = 0
        for b in batch_count:
            t = batch_encoder_out.pop(0)
            # in_encoder_out[st:st + b, 0:t.shape[1]] = t
            in_encoder_out_lens[st:st + b] = batch_encoder_lens.pop(0)
            for i in range(b):
                for j in range(beam_size):
                    cur_hyp = all_hyps.pop(0)
                    cur_time_stamp = all_time_stamp.pop(0)
                    cur_len = len(cur_hyp) + 2
                    in_hyp = [self.sos] + cur_hyp + [self.eos]
                    in_hyps_pad_sos_eos[st + i][j][0:cur_len] = in_hyp
                    in_hyps_lens_sos[st + i][j] = cur_len - 1
                    if len(in_all_time_stamp) == 0:
                        in_all_time_stamp.append([cur_time_stamp])
                    else :
                        if len(in_all_time_stamp[-1]) == beam_size:
                            in_all_time_stamp.append([cur_time_stamp])
                        else :
                            in_all_time_stamp[-1].append(cur_time_stamp)
                    if self.bidecoder:
                        r_in_hyp = [self.sos] + cur_hyp[::-1] + [self.eos]
                        in_r_hyps_pad_sos_eos[st + i][j][0:cur_len] = r_in_hyp
                    in_ctc_score[st + i][j] = all_ctc_score.pop(0)
            st += b
        in_encoder_out_lens = np.expand_dims(in_encoder_out_lens, axis=1)
        in_tensor_0 = pb_utils.Tensor("encoder_out", in_encoder_out)
        in_tensor_1 = pb_utils.Tensor("encoder_out_lens", in_encoder_out_lens)
        in_tensor_2 = pb_utils.Tensor("hyps_pad_sos_eos", in_hyps_pad_sos_eos)
        in_tensor_3 = pb_utils.Tensor("hyps_lens_sos", in_hyps_lens_sos)
        input_tensors = [in_tensor_0, in_tensor_1, in_tensor_2, in_tensor_3]
        if self.bidecoder:
            in_tensor_4 = pb_utils.Tensor("r_hyps_pad_sos_eos", in_r_hyps_pad_sos_eos)
            input_tensors.append(in_tensor_4)
        in_tensor_5 = pb_utils.Tensor("ctc_score", in_ctc_score)
        input_tensors.append(in_tensor_5)

        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names = ["score",'best_index'],
            inputs=input_tensors)

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            score = pb_utils.get_output_tensor_by_name(inference_response, 'score')
            best_index = pb_utils.get_output_tensor_by_name(inference_response, 'best_index')
            if best_index.is_cpu():
                best_index = best_index.as_numpy()
                score = score.as_numpy()
            else:
                best_index = from_dlpack(best_index.to_dlpack())
                best_index = best_index.cpu().numpy()
                score = from_dlpack(score.to_dlpack())
                score = score.cpu().numpy()
            # logger.info('best_index : {} '.format(best_index))
            logger.info('decoder end')
            hyps = []
            timestamps = []
            scores = []
            idx = 0
            for cands, cand_lens, time_stamps in zip(in_hyps_pad_sos_eos, in_hyps_lens_sos, in_all_time_stamp):
                best_idx = best_index[idx][0]
                best_score = score[idx][best_idx]
                best_cand_len = cand_lens[best_idx] - 1  # remove sos
                best_cand = cands[best_idx][1: 1 + best_cand_len].tolist()
                hyps.append(best_cand)
                idx += 1
                if debug:
                    logger.info('best_idx : {} '.format(best_idx))
                    logger.info('hyps: {} '.format(hyps))
                    logger.info('best_cand: {} '.format(best_cand))
                    logger.info('time_stamp: {} '.format(time_stamps[best_idx]))
                    logger.info('best cand text : {} '.format([self.vocab[i] for i in best_cand] ))
                    logger.info('score : {} '.format(score))

                frame_shift_in_ms = 40
                time_stamp_gap_ = 100
                time_stamp_list = []
                time_stamp = time_stamps[best_idx]
                for j in range(len(time_stamp)):
                    if time_stamp[j] * frame_shift_in_ms - time_stamp_gap_ > 0 :
                        start = time_stamp[j] * frame_shift_in_ms - time_stamp_gap_
                    else :
                        start = 0
                    if j > 0:
                        if (time_stamp[j] - time_stamp[j - 1]) * frame_shift_in_ms < time_stamp_gap_ :
                            start = (time_stamp[j-1] + time_stamp[j]) / 2 * frame_shift_in_ms
                    end = time_stamp[j] * frame_shift_in_ms
                    if j < len(time_stamp) - 1:
                        if (time_stamp[j + 1] - time_stamp[j]) * frame_shift_in_ms < time_stamp_gap_ :
                            end = (time_stamp[j + 1] + time_stamp[j]) / 2 * frame_shift_in_ms
                    time_stamp_dict = dict()
                    wd = self.vocab[best_cand[j]]
                    if re.search("[A-Z]",wd):
                        wd = p.sub('', wd)
                    time_stamp_dict["word"] = wd
                    time_stamp_dict["start"] = start
                    time_stamp_dict["end"] = end
                    time_stamp_list.append(time_stamp_dict)
                score_p = math.exp(best_score / len(best_cand))
                # logger.info('time_stamp_list: {} '.format(time_stamp_list))
                logger.info('best_score: {} avg_best_score: {} score_p:{}'.format(best_score, best_score / len(best_cand), score_p))
                timestamps.append(time_stamp_list)
                # scores.append(best_score / len(best_cand))
                scores.append(score_p)

            logger.info('time_stamp end')
            #logger.info('final hyps: {} '.format(hyps))
            hyps = map_batch(hyps, self.vocabulary,
                             min(multiprocessing.cpu_count(), len(in_ctc_score)))
            # logger.info('final hyps after: {} '.format(hyps))
            st = 0
            for b in batch_count:
                sents = np.array(hyps[st:st + b])
                # sents[0] = p.sub('', sents[0])
                # sents = [p.sub('', i) for i in sents]
                for idx in range(len(sents)):
                    sents[idx] = p.sub('', sents[idx])
                out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
                logger.info('sents : {} '.format(sents))

                timestamp = np.array(timestamps[st:st + b])
                out1 = pb_utils.Tensor("OUTPUT1", timestamp.astype(self.out1_dtype))
                # logger.info('timestamp : {} '.format(timestamp))

                score = np.array(scores[st:st + b])
                out2 = pb_utils.Tensor("OUTPUT2", score.astype(self.out2_dtype))
                logger.info('score: {} '.format(score))
                # inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
                inference_response = pb_utils.InferenceResponse(output_tensors=[out0, out1, out2])
                responses.append(inference_response)
                st += b
            logger.info('rec end')
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

