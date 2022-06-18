# QuoteR
Official code and data of the ACL 2022 paper "QuoteR: A Benchmark of Quote Recommendation for Writing" [[pdf](https://aclanthology.org/2022.acl-long.27.pdf)]



## 1. Requirements

* nltk==3.5
* numpy==1.19.5
* sklearn==0.0
* torch==1.7.1+cu110
* transformers==3.0.2
* OpenHowNet==0.0.1a11

## 2. Usage

###  2.1 Generate sememe data

`english_word_sememe.py` and `chinese_token_sememe.py` are used to generate the corresponding sememe data for English and Chinese, respectively.

### 2.2  Modify files in the Transformer Python library

Modify the `modeling_bert.py` file in the Transformer Python library and add three Classes (SememeEmbeddings, BertSememeEmbeddings, BertSememeModel) in `bert_chinese_sememe.py` or `bert_english_sememe.py`. 

Note that change the path of the corresponding sememe data in the SememeEmbeddings Class. Then add the BertSemeModel Class to the `__init__.py` file in the Transformer Python library.

### 2.3  Model training and testing

The files `english_train_test.py`, `modern_chinese_train_test.py` and `ancient_chinese_train_test.py` were used for training and testing, respectively.

## Cite

If the code or data help you, please cite the following paper.

```
@inproceedings{qi2022quoter,
  title={QuoteR: A Benchmark of Quote Recommendation for Writing},
  author={Qi, Fanchao and Yang, Yanhui and Yi, Jing and Cheng, Zhili and Liu, Zhiyuan and Sun, Maosong},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  url = {https://aclanthology.org/2022.acl-long.27},
  pages={336--348},
  year={2022}
}
```



