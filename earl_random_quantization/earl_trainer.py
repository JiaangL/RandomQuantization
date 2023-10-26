import os
import torch
import pickle as pkl
from dataset import OneShotIterator, KGETrainDataset
from torch.utils.data import DataLoader
from torch import optim
from trainer import Trainer
import torch.nn as nn
from model import MLP, ConRelEncoder, MulHopEncoder
import dgl
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_base import *
from utils import *


class EARLTrainer(Trainer):
    def __init__(self, args):
        super(EARLTrainer, self).__init__(args)

        self.rand_quant = args.random_entity_quantization
        self.num_step = args.num_step
        self.train_bs = args.train_bs
        self.lr = args.lr
        self.log_per_step = args.log_per_step
        self.check_per_step = args.check_per_step
        self.early_stop_patience = args.early_stop_patience

        self.train_iter = OneShotIterator(DataLoader(self.train_dataset,
                                                      batch_size=self.train_bs,
                                                      shuffle=True,
                                                      num_workers=max(1, args.cpu_num // 2),
                                                      collate_fn=KGETrainDataset.collate_fn))

        res_ent = pkl.load(open(os.path.join(args.data_path, f'res_ent_{self.args.res_ent_ratio}.pkl'), 'rb'))

        self.res_ent_map = res_ent['res_ent_map'].to(self.args.gpu)
        num_res_ent = self.res_ent_map.shape[0]
        self.res_ent_emb = nn.Parameter(torch.Tensor(num_res_ent, args.ent_dim).to(args.gpu))
        nn.init.xavier_uniform_(self.res_ent_emb, gain=nn.init.calculate_gain('relu'))

        self.topk_idx = res_ent['topk_idx'].to(self.args.gpu)
        self.topk_idx = self.topk_idx[:, :args.top_k]

        self.ent_sim = res_ent['topk_sim'].to(self.args.gpu)
        self.ent_sim = self.ent_sim[:, :args.top_k]
        self.ent_sim = torch.softmax(self.ent_sim/0.2, dim=-1)

        self.con_rel_encoder = ConRelEncoder(args).to(args.gpu)
        self.mul_hop_encoder = MulHopEncoder(args).to(args.gpu)
        self.mul_hop_encoder2 = MulHopEncoder(args).to(args.gpu)

        self.proj = MLP(args.ent_dim*2, args.ent_dim, args.ent_dim).to(args.gpu)

        self.token_emb = nn.Parameter(torch.Tensor(args.num_anchors, args.ent_dim).to(args.gpu))
        nn.init.xavier_uniform_(self.token_emb, gain=nn.init.calculate_gain('relu'))
        torch.manual_seed(31) 

        """ Randomly select *anchors* for quantization """
        if self.rand_quant:
            random_selection = torch.rand(args.num_ent, args.num_anchors)
            q = random_selection.kthvalue(args.top_k, dim=1).values
            mask = random_selection <= q.unsqueeze(1) # Each entity only has top-k anchors. Others are masked

        """ Compute entity code entropy """
        if args.code_level_distinguish:
            self.entropy = count_anchorset_entropy(mask)
            print(f'Entity Code Entropy: {self.entropy}')

        """ Compute entity code k-nn Jaccard Distance """
        if args.codeword_level_distinguish:
            topk_list = [200, 400, 600, 800, 1000]
            for topk in topk_list:
                anchorset_distance_topk = compute_anchorset_dist_topk(mask, args.gpu, topk=topk)
                print(f'Among top-{topk} neighbors, Entity Code Jaccard distance is: {anchorset_distance_topk}')

        """ Set codeword weights """
        if args.random_weights and args.equal_weights:
            raise Exception("Selected both random or equal codewrd weights. At most select one of them.")
        elif args.random_weights and self.rand_quant:
            codeword_weights = torch.rand(args.num_ent, args.num_anchors) # Random codeword weights (RW)
        elif args.equal_weigts and self.rand_quant:
            codeword_weights = torch.ones(args.num_ent, args.num_anchors) # Codeword weights all equal to 1 (EW)

        if self.rand_quant:
            self.codeword_weights = nn.Parameter(codeword_weights * mask.float(), requires_grad=False).to(args.gpu)

        self.rel_feat = nn.Parameter(torch.Tensor(args.num_rel, args.rel_dim).to(args.gpu), requires_grad=True)
        nn.init.xavier_uniform_(self.rel_feat, gain=nn.init.calculate_gain('relu'))
        self.entity_mlp = MLP(args.ent_dim, args.ent_dim, args.ent_dim, dropout_rate=args.mlp_dropout).to(args.gpu)

        # optimizer
        self.optimizer = optim.Adam(
                                    list(self.entity_mlp.parameters()) + list(self.mul_hop_encoder.parameters()) +
                                    [self.rel_feat] + [self.token_emb],
                                    lr=self.lr)

        self.cal_num_param()


    def cal_num_param(self):
        num_param = 0
        print('parameters:')
        print('entity-mlp parameters:')
        for name, param in self.entity_mlp.named_parameters():
            self.logger.info('\t{:45}\t{}\t{}'.format(name, param.size(), param.numel()))
            num_param += param.numel()

        print('gnn parameters:')
        for name, param in self.mul_hop_encoder.named_parameters():
            self.logger.info('\t{:45}\t{}\t{}'.format(name, param.size(), param.numel()))
            num_param += param.numel()

        name = 'token_emb'
        param = self.token_emb
        self.logger.info('\t{:45}\t{}\t{}'.format(name, param.size(), param.numel()))
        num_param += param.numel()

        self.logger.info(f'\ttotal: {num_param / 1e6} M')

        return num_param


    def get_emb(self):
        if self.rand_quant:
            """ Get entity embeddings by fully random entity quantization """
            cat_ent_emb = torch.mm(self.codeword_weights, self.token_emb)
            cat_ent_emb = self.entity_mlp(cat_ent_emb)
        else: # EARL original quantization strategies
            con_rel_info = self.con_rel_encode()
            kn_res_ent_info = self.kn_res_ent_encode()
            cat_ent_emb = self.proj(torch.cat([kn_res_ent_info, con_rel_info], dim=-1))
            cat_ent_emb[self.res_ent_map] = self.res_ent_emb
        
        """ Encode entity codes """
        ent_emb, rel_emb = self.mul_hop_encoder(self.train_g_bidir, cat_ent_emb)
        # rel_emb = self.rel_feat # rel_emb can also be got in this way. (won't hurt the performance)

        return ent_emb, rel_emb


    def train_one_step(self):
        # batch data
        batch = next(self.train_iter)
        pos_triple, neg_tail_ent, neg_head_ent = [b.to(self.args.gpu) for b in batch]

        # get ent and rel emb
        ent_emb, rel_emb = self.get_emb()

        # cal loss
        kge_loss = self.get_loss(pos_triple, neg_tail_ent, neg_head_ent, ent_emb, rel_emb)
        loss = kge_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, istest=False, num_cand='all'):
        if istest:
            dataloader = self.test_dataloader
        else:
            dataloader = self.valid_dataloader

        with torch.no_grad():
            # get ent and rel emb
            ent_emb, rel_emb = self.get_emb()

        results, count = self.get_rank(dataloader, ent_emb, rel_emb, num_cand)

        for k, v in results.items():
            results[k] = v / count

        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results

