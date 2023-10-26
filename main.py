import argparse
from earl_random_quantization.utils_base import init_dir
from earl_random_quantization.earl_trainer import EARLTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # task level setting
    parser.add_argument('--data_path', default='./data/wn18rr')
    parser.add_argument('--task_name', default='earl_rotate_wn18rr')
    parser.add_argument('--random_entity_quantization', default=True, type=bool)
    parser.add_argument('--random_weights', default=False, type=bool)
    parser.add_argument('--equal_weights', default=True, type=bool)
    parser.add_argument('--code_level_distinguish', default=False, type=bool)
    parser.add_argument('--codeword_level_distinguish', default=False, type=bool)

    # file setting
    parser.add_argument('--state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', default='./tb_log', type=str)

    # training setting
    parser.add_argument('--num_step', default=100000, type=int)
    parser.add_argument('--train_bs', default=1024, type=int) # default 1024
    parser.add_argument('--eval_bs', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=int)
    parser.add_argument('--log_per_step', default=10, type=int)
    parser.add_argument('--check_per_step', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=50, type=int) # default: 20
    parser.add_argument('--top_k', default=15, type=int) # set to (s + k) for fully random entity quantization
    parser.add_argument('--res_ent_ratio', default='0p1', type=str)

    # model setting
    parser.add_argument('--adv_temp', default=1, type=float)
    parser.add_argument('--gamma', default=10, type=float) # margin of scoring functions
    parser.add_argument('--dim', default=100, type=int) # WN18RR: 200; FB15k-237: 150; codex: 100
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_rel', default=None)
    parser.add_argument('--num_ent', default=None)
    parser.add_argument('--num_anchors', default=7795, type=int) # 1450, 4055, and 7795 for FB15k-237, WN18RR, and CoDEx-L.
    parser.add_argument('--mlp_dropout', default=0, type=int)
    parser.add_argument('--hid_dropout', default=0, type=int)

    # device
    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument('--cpu_num', default=10, type=int)

    # tracker
    parser.add_argument('--wandb', default=False, type=bool)

    args = parser.parse_args()
    init_dir(args)

    # dim for RotatE
    args.ent_dim = args.dim * 2
    args.rel_dim = args.dim

    # gourp and run name for WANDB
    dataset_name = args.data_path.split("/")[-1]
    if dataset_name == 'wn18rr':
        dataset_name = 'WN'
        args.dim = 200
        args.num_anchors = 4055
        args.tags = ['WN18RR']
    elif dataset_name == 'FB15k-237':
        dataset_name = 'FB'
        args.dim = 150
        args.num_anchors = 1450
        args.tags = ['FB15k-237']
    elif dataset_name == 'codex-l':
        dataset_name = 'codex'
        args.tags = ['codex-l']
    args.group = None
    args.run_name = dataset_name

    trainer = EARLTrainer(args)

    trainer.train()

