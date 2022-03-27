from transformers import BertTokenizer
import numpy as np
import OpenHowNet


# get all chinese semems
PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
hownet_dict = OpenHowNet.HowNetDict()
all_semems = hownet_dict.get_all_sememes()
print("all semems: ", len(all_semems))

# get bert-base-chinese vocab
vocab = tokenizer.vocab
print(len(vocab))
all_tokens = [token for token, idx in vocab.items()]
print("all tokens: ", len(all_tokens))


# Generate sememes for each token
sememe_onehot = np.zeros((len(all_tokens), len(all_semems)))
for i in range(len(all_tokens)):
    semems = hownet_dict.get_sememes_by_word(all_tokens[i], structured=False, lang="zh", merge=True)
    print(semems)
    if len(semems) > 0:
        word2semem_idx = []
        word2sememe = [s for s in semems]
        for s in word2sememe:
            if s in all_semems:
                sememe_onehot[i][all_semems.index(s)] = 1
print("word2semem one hot: ", sememe_onehot.shape)
np.save("./data/bert_chinese_token_sememe.npy", sememe_onehot)