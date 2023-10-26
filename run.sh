dataset='./data/wn18rr'
task_name='earl_rotate_wn18rr'
dim=200
gpu=0
rq=True
ew=True # Equal codeword weights
entropy=False
jaccard_distance=False
wandb=False

python main.py --data_path ${dataset} --task_name ${task_name} --random_entity_quantization ${rq} --equal_weights ${ew}  --code_level_distinguish ${entropy} --codeword_level_distinguish ${jaccard_distance} --dim ${dim} --gpu cuda:${gpu} --wandb ${wandb}