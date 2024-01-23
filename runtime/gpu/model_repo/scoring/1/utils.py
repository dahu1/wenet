import math
import torch
import concurrent.futures

from collections import defaultdict
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pad_sequence
import copy
import random, re

# from wenet.transformer.context_module import ContextModule
# from wenet.transformer.ctc import CTC

import logging
logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger('streaming-interface')
logger.setLevel(logging.INFO)

def celue(ss):
    rand_x = random.randint(0,100)/100
    if rand_x <= 0.9 and not re.search('[吗么啥]',ss[-2:]):
        ss = re.sub("？$","。",ss)
    return ss

def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp

def __tokenize_by_bpe_model(sp, txt):
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r'([\u4e00-\u9fff])')
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            for p in sp.encode_as_pieces(ch_or_w):
                tokens.append(p)

    return tokens


def tokenize_list(context_lists, symbol_table, bpe_model=None):
    """ Read biasing list from the biasing list address, tokenize and convert it
        into token id
    """
    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    else:
        sp = None

    context_list = []
    for context_txt in context_lists:
        labels = []
        tokens = []
        if bpe_model is not None:
            tokens = __tokenize_by_bpe_model(sp, context_txt)
        else:
            for ch in context_txt:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)
        for ch in tokens:
            if ch in symbol_table:
                labels.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                labels.append(symbol_table['<unk>'])
        context_list.append(labels)
    return context_list

def tokenize_dict(context_dict, symbol_table, bpe_model=None):
    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    else:
        sp = None

    context_list = []
    for context_txt, graph_score in context_dict.items():

        labels = []
        tokens = []
        if bpe_model is not None:
            tokens = __tokenize_by_bpe_model(sp, context_txt)
        else:
            for ch in context_txt:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)
        for ch in tokens:
            if ch in symbol_table:
                labels.append(symbol_table[ch])
                # labels.append(ch)
            elif '<unk>' in symbol_table:
                labels.append(symbol_table['<unk>'])
                # labels.append('<unk>')
        item = {"key": labels, "value": graph_score}
        context_list.append(item)
    return context_list

def tbbm(sp, context_txt):
    return __tokenize_by_bpe_model(sp, context_txt)

def ctc_prefix_beam_search(
    encoder_out, beam_size, ctc_probs, context_graph = None) :
    # encoder_out  # (maxlen, encoder_dim)

    maxlen = encoder_out.size(0)
    # ctc_probs = ctc_probs.squeeze(0)
    # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score,
    #                       context_state, context_score))
    # cur_hyps = [(tuple(), (0.0, -float('inf'), 0, 0.0))]
    cur_hyps = [(tuple(), (0.0, -float('inf'), 0, 0.0, 0.0, 0.0,[],[],-float('inf')))]
    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        logp = ctc_probs[t]  # (vocab_size,)
        abs_time_step = t
        # key: prefix, value (pb, pnb, context_state, context_score, vb, vnb, tb, tnb, cur_token_prob),
        # default value(-inf, -inf, 0, 0.0)
        # next_hyps = defaultdict(lambda: (-float('inf'), -float('inf'), 0, 0.0))
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf'), 0, 0.0, -float('inf'),-float('inf'),[],[],-float('inf')))
        # 2.1 First beam prune: select topk best
        top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
        for s in top_k_index:
            s = s.item()
            ps = logp[s].item()
            for prefix, (pb, pnb, c_state, c_score, vb, vnb, tb, tnb, ctp) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:  # blank
                    n_pb, n_pnb, _, _, n_vb, n_vnb, n_tb, n_tnb, n_ctp = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    n_vb = max(vb, vnb) + ps        # viterbi_score()
                    n_tb = tb if vb > vnb else tnb  # times()
                    n_tb = copy.deepcopy(n_tb)
                    next_hyps[prefix] = (n_pb, n_pnb, c_state, c_score, n_vb, n_vnb, n_tb, n_tnb, n_ctp)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb, _, _, n_vb, n_vnb, n_tb, n_tnb, n_ctp = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    if n_vnb < vnb + ps:
                        n_vnb = vnb + ps
                        if n_ctp < ps:
                            n_ctp = ps
                            n_tnb = tnb
                            assert len(n_tnb) > 0
                            n_tnb = copy.deepcopy(n_tnb)
                            n_tnb[-1] = abs_time_step
                    next_hyps[prefix] = (n_pb, n_pnb, c_state, c_score, n_vb, n_vnb, n_tb, n_tnb, n_ctp)

                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb, _, _, n_vb, n_vnb, n_tb, n_tnb, n_ctp = next_hyps[n_prefix]
                    new_c_state, new_c_score = 0, 0
                    if context_graph is not None and context_graph.graph_biasing:
                        new_c_state, new_c_score = context_graph. \
                            find_graph_dict_next_state(c_state, s)
                            # find_next_state(c_state, s)
                    n_pnb = log_add([n_pnb, pb + ps])
                    if n_vnb < vb + ps:
                        n_vnb = vb + ps
                        n_ctp = ps
                        n_tnb = tb
                        n_tnb = copy.deepcopy(n_tnb)
                        n_tnb.append(abs_time_step)
                        # logger.info('case 2 abs_time_step: {} len(n_tnb): {}'.format(abs_time_step, len(n_tnb)))
                    next_hyps[n_prefix] = (n_pb, n_pnb, new_c_state, c_score + new_c_score, n_vb, n_vnb, n_tb, n_tnb, n_ctp)
                else:
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb, _, _, n_vb, n_vnb, n_tb, n_tnb, n_ctp = next_hyps[n_prefix]
                    new_c_state, new_c_score = 0, 0
                    if context_graph is not None and context_graph.graph_biasing:
                        new_c_state, new_c_score = context_graph. \
                            find_graph_dict_next_state(c_state, s)
                            # find_next_state(c_state, s)
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    # logger.info('n_vnb: {} vb: {} vnb: {} ps: {}'.format(n_vnb, vb, vnb, ps))
                    if n_vnb < max(vb, vnb) + ps:
                        n_vnb = max(vb, vnb) + ps     # viterbi_score()
                        n_ctp = ps
                        n_tnb = tb if vb > vnb else tnb  #times()
                        n_tnb = copy.deepcopy(n_tnb)
                        n_tnb.append(abs_time_step)
                        # logger.info('case 3 abs_time_step: {} len(n_tnb): {}'.format(abs_time_step, len(n_tnb)))
                    next_hyps[n_prefix] = (n_pb, n_pnb, new_c_state, c_score + new_c_score, n_vb, n_vnb, n_tb, n_tnb, n_ctp)

        # 2.2 Second beam prune
        next_hyps = sorted(
            next_hyps.items(),
            key=lambda x: log_add([x[1][0], x[1][1]]) + x[1][3],
            reverse=True)
        cur_hyps = next_hyps[:beam_size]
    # hyps = [(log_add([y[1][0], y[1][1]]) + y[1][3], y[0])
    #         for y in cur_hyps]
    hyps = [(log_add([y[1][0], y[1][1]]) + y[1][3], y[0], y[1][6] if y[1][4] > y[1][5] else y[1][7]) for y in cur_hyps]
    return hyps

def ctc_prefix_beam_search_batch(in_encoder_out, beam_size, in_ctc_log_probs, online_graph_list):
    n = len(in_encoder_out)
    with concurrent.futures.ThreadPoolExecutor(max_workers = n) as executor:
        # 任务队列
        res = []

        # 任务调度
        for i in range(n):
            encoder_out = torch.from_numpy(in_encoder_out[i])
            ctc_log_probs = torch.from_numpy(in_ctc_log_probs[i])
            context_graph = online_graph_list[i]
            res.append(executor.submit(ctc_prefix_beam_search, encoder_out, beam_size, ctc_log_probs, context_graph))

        # 获取解码结果
        batch_results = []
        for future in res:
            batch_results.append(future.result())
    return batch_results

def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table

class ContextGraph:
    """ Context decoding graph, constructing graph using dict instead of WFST
        Args:
            context_list_path(str): context list path
            bpe_model(str): model for english bpe part
    """
    def __init__(self):
        self.graph1 = {0: {}}
        self.graph_size1 = 0
        self.state2token1 = {}
        self.back_score1 = {0: 0.0}
        self.state_score1 = {0: 0}

        self.hot_word_dict = {}
        self.filter_hot_word_dict = dict()

        self.graph_biasing = True
        self.deep_biasing = True
        self.deep_biasing_score = 1.0
        self.context_filtering = True
        self.filter_threshold = -4.0


        # self.context_dict = {'你好': 1, '天气': 2}
        self.context_dict = dict()

    def context_tonkenize(self, symbol_table, bpe_model = None):
        self.context_list = tokenize_list(self.context_dict.keys(), symbol_table, bpe_model)
        self.context_list1 = tokenize_dict(self.context_dict, symbol_table, bpe_model)

    def load_context_path(self, context_list_path):
        for line in  open(context_list_path, "r", encoding="utf-8"):
            lin = line.strip().split("\t")
            assert len(lin) == 2
            k = lin[0]
            v = float(lin[1])
            self.context_dict[k] = v

    def build_graph_dict(self, context_list1):
        for item in context_list1 :
            context_token = item["key"]
            graph_score = item["value"]
            now_state = 0
            for i in range(len(context_token)):
                if context_token[i] in self.graph1[now_state]:
                    now_state = self.graph1[now_state][context_token[i]]
                    self.state_score1[now_state] = max(graph_score, self.state_score1[now_state])
                    if i == len(context_token) - 1:
                        self.back_score1[now_state] = 0
                    else:
                        if self.back_score1[now_state] !=0 :
                            self.back_score1[now_state] = -(i + 1) * self.state_score1[now_state]
                else:
                    self.graph_size1 += 1
                    self.graph1[self.graph_size1] = {}
                    self.graph1[now_state][context_token[i]] = self.graph_size1
                    now_state = self.graph_size1
                    if i != len(context_token) - 1:
                        self.back_score1[now_state] = -(i + 1) * graph_score
                    else:
                        self.back_score1[now_state] = 0
                    self.state2token1[now_state] = context_token[i]
                    self.state_score1[now_state] = graph_score
        # logger.info('self.graph1: {} self.graph_size1: {}'.format(self.graph1, self.graph_size1))
        # logger.info('self.back_score1: {}'.format(self.back_score1))
        # logger.info('self.state2token1: {}'.format(self.state2token1))
        # logger.info('self.state_score1: {}'.format(self.state_score1))

    def find_graph_dict_next_state(self, now_state: int, token: int):
        if token in self.graph1[now_state]:
            state = self.graph1[now_state][token]
            return self.graph1[now_state][token], self.state_score1[state]
        back_score = self.back_score1[now_state]
        now_state = 0
        if token in self.graph1[now_state]:
            state = self.graph1[now_state][token]
            return self.graph1[now_state][token], \
                back_score + self.state_score1[state]
        return 0, back_score


    def get_context_list_tensor(self, context_list: List[List[int]]):
        """Add 0 as no-bias in the context list and obtain the tensor
           form of the context list
        """
        context_list_tensor = [torch.tensor([0], dtype=torch.int32)]
        for context_token in context_list:
            context_list_tensor.append(torch.tensor(context_token, dtype=torch.int32))
        context_list_lengths = torch.tensor([x.size(0) for x in context_list_tensor],
                                            dtype=torch.int32)
        context_list_tensor = pad_sequence(context_list_tensor,
                                           batch_first=True,
                                           padding_value=-1)
        return context_list_tensor, context_list_lengths
        # return context_list_tensor.numpy(), context_list_lengths.numpy()

    def two_stage_filtering(self,
                            context_list: List[List[int]],
                            ctc_posterior: torch.Tensor,
                            filter_window_size: int = 64):
        """Calculate PSC and SOC for context phrase filtering,
           refer to: https://arxiv.org/abs/2301.06735
        """
        if len(context_list) == 0:
            return context_list

        SOC_score = {}
        for t in range(1, ctc_posterior.shape[0]):
            if t % (filter_window_size // 2) != 0 and t != ctc_posterior.shape[0] - 1:
                continue
            # calculate PSC
            PSC_score = {}
            max_posterior, _ = torch.max(ctc_posterior[max(0,
                                         t - filter_window_size):t, :],
                                         dim=0, keepdim=False)
            max_posterior = max_posterior.tolist()
            for i in range(len(context_list)):
                score = sum(max_posterior[j] for j in context_list[i]) \
                    / len(context_list[i])
                PSC_score[i] = max(SOC_score.get(i, -float('inf')), score)
            PSC_filtered_index = []
            for i in PSC_score:
                if PSC_score[i] > self.filter_threshold:
                    PSC_filtered_index.append(i)
            if len(PSC_filtered_index) == 0:
                continue
            filtered_context_list = []
            for i in PSC_filtered_index:
                filtered_context_list.append(context_list[i])

            # calculate SOC
            win_posterior = ctc_posterior[max(0, t - filter_window_size):t, :]
            win_posterior = win_posterior.unsqueeze(0) \
                .expand(len(filtered_context_list), -1, -1)
            select_win_posterior = []
            for i in range(len(filtered_context_list)):
                select_win_posterior.append(torch.index_select(
                    win_posterior[0], 1,
                    torch.tensor(filtered_context_list[i],
                                 device=ctc_posterior.device)).transpose(0, 1))
            select_win_posterior = \
                pad_sequence(select_win_posterior,
                             batch_first=True).transpose(1, 2).contiguous()
            dp = torch.full((select_win_posterior.shape[0],
                             select_win_posterior.shape[2]),
                            -10000.0, dtype=torch.float32,
                            device=select_win_posterior.device)
            dp[:, 0] = select_win_posterior[:, 0, 0]
            for win_t in range(1, select_win_posterior.shape[1]):
                temp = dp[:, :-1] + select_win_posterior[:, win_t, 1:]
                idx = torch.where(temp > dp[:, 1:])
                idx_ = (idx[0], idx[1] + 1)
                dp[idx_] = temp[idx]
                dp[:, 0] = \
                    torch.where(select_win_posterior[:, win_t, 0] > dp[:, 0],
                                select_win_posterior[:, win_t, 0], dp[:, 0])
            for i in range(len(filtered_context_list)):
                SOC_score[PSC_filtered_index[i]] = \
                    max(SOC_score.get(PSC_filtered_index[i], -float('inf')),
                        dp[i][len(filtered_context_list[i]) - 1]
                        / len(filtered_context_list[i]))
        filtered_context_list = []
        for i in range(len(context_list)):
            if SOC_score.get(i, -float('inf')) > self.filter_threshold:
                filtered_context_list.append(context_list[i])
        return filtered_context_list
