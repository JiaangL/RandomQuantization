# RandomQuantization
Release the code of
[Random Entity Quantization for Parameter-Efficient Compositional Knowledge Graph Representation](https://arxiv.org/abs/2310.15797#). 
This paper has been accepted by **EMNLP 2023** main conference.

<div  align="center">  
<img src="./EntityQuantization.png" width = "600" height = "375" alt="EntityQuantization" align=center />
</div>  

## Preparation
### Enviroment
The code is tested under ```torch==1.12.0``` and ```dgl==1.0.0```. The requirements of specific version is not very strict. Run with no bugs, then you are set.
### Data
Datasets we used are in ```./data```. Unzip the files before using them. If you want to run without the random entity quantization and test the original EARL quantization strategy, please use ```pre_process.ipynb``` to process the data.

## Run
Run the random entity quantization by running ```bash run.sh```.

In this script, you can open ```--code_level_distinguish``` and ```--codeword_level_distinguish``` to view the entropy and nearest neighbor Jaccard distance of the entity codes. Experiments are tracked by [WandB](https://wandb.ai/site) if setting ```--wandb True```.

## Acknowledgement
This repo benifits from [NodePiece](https://github.com/migalkin/NodePiece) and [EARL](https://github.com/zjukg/EARL). Thanks for their wonderful works.

## Contact and Citations
Feel free to leave issues or [contact us](mailto:jali@mail.ustc.edu.cn) if you have any questions.
If you find our paper or code useful, please cite our paper as:
```
@inproceedings{li-etal-2023-random,
    title = "Random Entity Quantization for Parameter-Efficient Compositional Knowledge Graph Representation",
    author = "Li, Jiaang  and
      Wang, Quan  and
      Liu, Yi  and
      Zhang, Licheng  and
      Mao, Zhendong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.177",
    doi = "10.18653/v1/2023.emnlp-main.177",
    pages = "2917--2928",
    abstract = "Representation Learning on Knowledge Graphs (KGs) is essential for downstream tasks. The dominant approach, KG Embedding (KGE), represents entities with independent vectors and faces the scalability challenge. Recent studies propose an alternative way for parameter efficiency, which represents entities by composing entity-corresponding codewords matched from predefined small-scale codebooks. We refer to the process of obtaining corresponding codewords of each entity as entity quantization, for which previous works have designed complicated strategies. Surprisingly, this paper shows that simple random entity quantization can achieve similar results to current strategies. We analyze this phenomenon and reveal that entity codes, the quantization outcomes for expressing entities, have higher entropy at the code level and Jaccard distance at the codeword level under random entity quantization. Therefore, different entities become more easily distinguished, facilitating effective KG representation. The above results show that current quantization strategies are not critical for KG representation, and there is still room for improvement in entity distinguishability beyond current strategies.",
}
```
