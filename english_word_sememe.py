import torch
from transformers import BertTokenizer
import numpy as np
import OpenHowNet
import nltk


# load english dataset
def load_data(path):
    # return words list and labels
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip().lower().split('\t') for line in lines]
        train_former = [line[0] for line in lines[:101171]]
        train_quote = [line[1] for line in lines[:101171]]
        train_latter = [line[2] for line in lines[:101171]]
        valid_former = [line[0] for line in lines[101171:113942]]
        valid_quote = [line[1] for line in lines[101171:113942]]
        valid_latter = [line[2] for line in lines[101171:113942]]
        test_former = [line[0] for line in lines[113942:]]
        test_quote = [line[1] for line in lines[113942:]]
        test_latter = [line[2] for line in lines[113942:]]
        all_quotes = train_quote + valid_quote + test_quote
    all_quotes = list(set(all_quotes))
    all_quotes.sort()
    y_train = [all_quotes.index(q) for q in train_quote]
    y_valid = [all_quotes.index(q) for q in valid_quote]
    y_test = [all_quotes.index(q) for q in test_quote]

    return train_former, train_latter, train_quote, valid_former, valid_latter, valid_quote, test_former, test_latter, test_quote, torch.LongTensor(
        y_train), torch.LongTensor(y_valid), torch.LongTensor(
            y_test), all_quotes


print("loading dataset......")
data_path = "./data/english.txt"
train_former, train_latter, train_quote, valid_former, valid_latter, valid_quote, test_former, test_latter, test_quote, y_train, y_valid, y_test, all_quotes = load_data(
    data_path)
print("tran  valid  test:", len(train_former), len(valid_former), len(test_former))
print("all quotes: ", len(all_quotes))
print("train quote:", len(list(set(train_quote))))
print("valid quote:", len(list(set(valid_quote))))
print("test quote:", len(list(set(test_quote))))

# load pretrained model
PRETRAINED_MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
hownet_dict = OpenHowNet.HowNetDict()

all_quote_words = []
for s in all_quotes:
    q_words = [w for w in nltk.word_tokenize(s)]
    all_quote_words.extend(q_words)
all_quote_words = list(set(all_quote_words))
all_quote_words.sort()
all_quote_words.insert(0, '<INS>')
all_quote_words.insert(101, '<INS>')
all_quote_words.insert(102, '<INS>')
print("all quote words ", len(all_quote_words))
print(all_quote_words[0])
print(all_quote_words[101])
print(all_quote_words[102])

# get bert-base-uncased vocab
vocab = tokenizer.vocab
print("vocab size: ", len(vocab))
all_tokens = [token for token, idx in vocab.items()]
print("all vocab token: ", len(all_tokens))

# get all english sememes
all_semems = []
for i in range(len(all_quote_words)):
    semems = hownet_dict.get_sememes_by_word(all_quote_words[i], structured=False, lang="en", merge=True)
    for s in semems:
        all_semems.append(s)
all_semems = list(set(all_semems))
all_semems.sort()
print("all sememes: ", len(all_semems))

# Generate sememes for each word
sememe_onehot = np.zeros((len(all_quote_words), len(all_semems)))
for i in range(len(all_quote_words)):
    semems = hownet_dict.get_sememes_by_word(all_quote_words[i], structured=False, lang="en", merge=True)
    print(semems)
    if len(semems) > 0:
        word2sememe = [s for s in semems]
        for s in word2sememe:
            sememe_onehot[i][all_semems.index(s)] = 1
print("word2sememe one hot: ", sememe_onehot.shape)
np.save("./data/english_word_sememe.npy", sememe_onehot)

# Generate the word index for each quote
all_word_ids = []
all_input_ids = []
for quote in all_quotes:
    quote_words = [w for w in nltk.word_tokenize(quote)]
    token2word = []
    for w in quote_words:
        tokens = tokenizer.tokenize(w)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        for id in ids:
            token2word.append([id, all_quote_words.index(w)])

    encoded_dict = tokenizer.encode_plus(quote,
                                         add_special_tokens=True,
                                         max_length=80,
                                         pad_to_max_length=True,
                                         truncation=True,
                                         return_attention_mask=True,
                                         return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    all_input_ids.append(input_ids)
    input_ids = input_ids.squeeze().tolist()
    word_ids = []
    for id in input_ids:
        if id == 101:
            word_ids.append(id)
        elif id == 102:
            word_ids.append(id)
        elif id == 0:
            word_ids.append(id)
        else:
            for i in range(len(token2word)):
                if id == token2word[i][0]:
                    word_ids.append(token2word[i][1])
                    token2word = token2word[i+1:]
                    break
    if len(word_ids) != 80:
        word_ids.extend(0 for _ in range(80-len(word_ids)))
    all_word_ids.append(torch.LongTensor(word_ids))

all_word_ids = torch.stack(all_word_ids, dim=0)
print("all word ids: ", all_word_ids.shape)
all_input_ids = torch.cat(all_input_ids, dim=0)
print("all input ids: ", all_input_ids.shape)

torch.save(all_word_ids, "./data/bert_base_uncased_quote2word.pt")