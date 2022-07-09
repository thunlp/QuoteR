import torch
from transformers import BertTokenizer, BertModel, AdamW, BertSememeModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import time
import random
from sklearn.metrics import accuracy_score
import numpy as np

random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = 8
sample_num = 19
learning_rate = 3e-5


# loading dataset
def load_data(path):
    # return words list and labels
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        train_former = [line[0] for line in lines[:93031]]
        train_quote = [line[1] for line in lines[:93031]]
        train_latter = [line[2] for line in lines[:93031]]
        valid_former = [line[0] for line in lines[93031:104784]]
        valid_quote = [line[1] for line in lines[93031:104784]]
        valid_latter = [line[2] for line in lines[93031:104784]]
        test_former = [line[0] for line in lines[104784:]]
        test_quote = [line[1] for line in lines[104784:]]
        test_latter = [line[2] for line in lines[104784:]]
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
data_path = "./data/ancient_chinese.txt"
train_former, train_latter, train_quote, valid_former, valid_latter, valid_quote, test_former, test_latter, test_quote, y_train, y_valid, y_test, all_quotes = load_data(
    data_path)
print("tran  valid  test:", len(train_quote), len(valid_quote),
      len(test_quote))
print("train quote:", len(list(set(train_quote))))
print("valid quote:", len(list(set(valid_quote))))
print("test quote:", len(list(set(test_quote))))

# get the Tokenizer used for pretraining model
PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)


def make_context_tensors(former, latter):
    input_ids = []
    token_type_ids = []
    attention_masks = []
    mask_ids = []
    for f, l in zip(former, latter):
        sent = f + "[MASK]" + l
        encoded_dict = tokenizer.encode_plus(sent,
                                             add_special_tokens=True,
                                             max_length=100,
                                             pad_to_max_length=True,
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        mask_index = encoded_dict['input_ids'][0].tolist().index(103)
        mask_ids.append(mask_index)
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, token_type_ids, attention_masks, torch.LongTensor(
        mask_ids)


print("loading train and valid data......")
train_input_ids, train_token_type_ids, train_attention_masks, train_mask_ids = make_context_tensors(
    train_former, train_latter)
valid_input_ids, valid_token_type_ids, valid_attention_masks, valid_mask_ids = make_context_tensors(
    valid_former, valid_latter)
print("train bert input:")
print(train_input_ids.shape, train_token_type_ids.shape,
      train_attention_masks.shape, train_mask_ids.shape)
print("valid bert input:")
print(valid_input_ids.shape, valid_token_type_ids.shape,
      valid_attention_masks.shape, valid_mask_ids.shape)


# Dataset and DataLoader
class Dataset(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_masks, mask_ids,
                 quote):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_masks = attention_masks
        self.mask_ids = mask_ids
        self.quote = quote

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.quote is None:
            return self.input_ids[idx], self.token_type_ids[
                idx], self.attention_masks[idx], self.mask_ids[idx]
        return self.input_ids[idx], self.token_type_ids[
            idx], self.attention_masks[idx], self.mask_ids[idx], self.quote[
                idx]


print("loading train and valid dataloader ...")
train_dataset = Dataset(input_ids=train_input_ids,
                        token_type_ids=train_token_type_ids,
                        attention_masks=train_attention_masks,
                        mask_ids=train_mask_ids,
                        quote=train_quote)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch,
                          shuffle=True,
                          num_workers=2)
valid_dataset = Dataset(input_ids=valid_input_ids,
                        token_type_ids=valid_token_type_ids,
                        attention_masks=valid_attention_masks,
                        mask_ids=valid_mask_ids,
                        quote=valid_quote)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=batch,
                          shuffle=True,
                          num_workers=2)


#  generat negative examples according to num
def generate_quotes(quote, num):
    quotes_selcet = all_quotes[:]
    quotes_selcet.remove(quote)
    quotes = random.sample(quotes_selcet, num)
    quotes.append(quote)
    random.shuffle(quotes)
    return quotes


def make_quote_tensors(quote):
    quotes = generate_quotes(quote, num=sample_num)
    label = quotes.index(quote)
    input_ids = []
    for q in quotes:
        encoded_dict = tokenizer.encode_plus(q,
                                             add_special_tokens=True,
                                             max_length=60,
                                             pad_to_max_length=True,
                                             truncation=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
    input_ids = torch.cat(input_ids, 0)  # [num, 60]
    return input_ids, label


# Define network
class Context_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.5)

    def forward(self, context_input_ids, context_token_type_ids,
                context_attention_masks, mask_ids):
        outputs = self.bert_model(input_ids=context_input_ids,
                                  token_type_ids=context_token_type_ids,
                                  attention_mask=context_attention_masks)
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_context = []
        for i in range(len(last_hidden_state)):
            hidden_state = last_hidden_state[i]  # [sequence_length, hidden_size]
            mask = hidden_state[mask_ids[i]]
            mask = self.dropout(mask)
            context = mask.unsqueeze(dim=0)  # context: [1, hidden_size]
            all_context.append(context)
        all_context = torch.cat(all_context, dim=0)  # all_context: [batch, hidden_size]
        return all_context


class Quote_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertSememeModel.from_pretrained(
            PRETRAINED_MODEL_NAME)

    def forward(self, quotes):
        quote_tensor = []
        labels = []
        for quote in quotes:
            quote_input_ids, label = make_quote_tensors(quote)
            quote_input_ids = quote_input_ids.to(device)
            outputs = self.bert_model(input_ids=quote_input_ids)
            last_hidden_state = outputs[0]  # [num_quotes, sequence_length, hidden_size]
            output = torch.mean(last_hidden_state, dim=1)  # [num_quotes, hidden_size]
            quote_tensor.append(output)
            labels.append(label)
        quote_tensor = torch.stack(quote_tensor, dim=0)  # [batch, num_quotes, hidden_size]
        return quote_tensor, labels


class QuotRec_Net(nn.Module):
    def __init__(self, contex_model, quote_model):
        super().__init__()
        self.contex_model = contex_model
        self.quote_model = quote_model

    def forward(self, input_ids, token_type_ids, attention_masks, mask_ids,
                quotes):
        # context_output: [batch, hidden_size]
        context_output = self.contex_model(input_ids, token_type_ids,
                                           attention_masks, mask_ids)
        context_output = context_output.unsqueeze(dim=1)  # [batch, 1, hidden_size]
        # quote_output: [batch, num, hidden_size]  labels: [batch]
        quote_output, labels = self.quote_model(quotes)
        quote_output = quote_output.permute(0, 2, 1)
        outputs = torch.matmul(context_output, quote_output).squeeze(
            dim=1)  # output: [batch, num_quotes]
        return outputs, torch.LongTensor(labels)


print("loading model......")
contex_model = Context_Encoder()
quote_model = Quote_Encoder()
model = QuotRec_Net(contex_model, quote_model)
model.to(device)


def training(model, epoch, train, valid, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(
        total, trainable))
    t_batch = len(train)
    v_batch = len(valid)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_acc = 0
    count = 0
    for epoch in range(epoch):
        start = time.clock()
        total_loss, total_acc = 0, 0
        print("epoch: ", epoch + 1)
        # train
        for i, (input_ids, token_type_ids, attention_masks, mask_ids,
                quotes) in enumerate(train):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_masks = attention_masks.to(device)
            mask_ids = mask_ids.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs, labels = model(input_ids, token_type_ids, attention_masks,
                                    mask_ids, quotes)
            labels = labels.to(device, dtype=torch.long)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(outputs.cpu().data, 1)
            acc = accuracy_score(pred, labels.cpu())
            total_loss += loss.item()
            total_acc += acc
        print('Train | Loss:{:.5f} Acc:{:.3f}'.format(total_loss,
                                                      total_acc / t_batch))
        # validation
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (input_ids, token_type_ids, attention_masks, mask_ids,
                    quotes) in enumerate(valid):
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_masks = attention_masks.to(device)
                mask_ids = mask_ids.to(device, dtype=torch.long)
                outputs, labels = model(input_ids, token_type_ids,
                                        attention_masks, mask_ids, quotes)
                labels = labels.to(device, dtype=torch.long)
                loss = criterion(outputs, labels)
                _, pred = torch.max(outputs.cpu().data, 1)
                acc = accuracy_score(pred, labels.cpu())
                total_loss += loss.item()
                total_acc += acc
            print('Valid | Loss:{:.5f} Acc:{:.3f}'.format(
                total_loss, total_acc / v_batch))
            if total_acc > best_acc:
                best_acc = total_acc
                if os.path.exists("./model"):
                    pass
                else:
                    os.mkdir("./model")
                torch.save(
                    model.quote_model.state_dict(), "./model/ancient_chinese_quote.pth")
                torch.save(
                    model.contex_model.state_dict(), "./model/ancient_chinese_context.pth")
                print('saving model with Acc {:.3f} '.format(total_acc / v_batch))
                count = 0
            else:
                count += 1
        model.train()
        end = time.clock()
        print('epoch running time:{:.0f}s'.format(end - start))
        # early stopping
        if count == 3:
            break


training(model=model,
         epoch=40,
         train=train_loader,
         valid=valid_loader,
         device=device)


# make quotes to bert tensor
def make_tensors(quotes):
    input_ids = []
    for q in quotes:
        encoded_dict = tokenizer.encode_plus(q,
                                             add_special_tokens=True,
                                             max_length=60,
                                             pad_to_max_length=True,
                                             truncation=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
    input_ids = torch.cat(input_ids, 0)
    return input_ids


quote_input_ids = make_tensors(all_quotes)
print("quote bert input:")
print(quote_input_ids.shape)

# Generate sentence vector for quotes
quote_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
model_dict = quote_model.state_dict()
save_model_state = torch.load("./model/ancient_chinese_quote.pth")
state_dict = {k[11:]: v for k, v in save_model_state.items() if k[11:] in model_dict.keys()}
model_dict.update(state_dict)
quote_model.load_state_dict(model_dict)
quote_model = quote_model.to(device)
quote_input_ids = quote_input_ids.to(device)

quote_embeddings = []
quote_model.eval()
with torch.no_grad():
    i = 0
    for input_ids in quote_input_ids:
        i += 1
        input_ids = input_ids.unsqueeze(dim=0)
        outputs = quote_model(input_ids=input_ids)
        hidden_states = outputs[0]  # hidden_states:[batch_size, sequence_length, hidden_size]
        quote_tensor = torch.mean(hidden_states, dim=1)  # quote_tensor: [batch_size, hidden_size]
        quote_embeddings.append(quote_tensor)
    quote_embeddings = torch.cat(quote_embeddings, dim=0)
print("quote tensor:")
print(quote_embeddings.shape)


# Use the mask method for training
class QuotRecNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, token_type_ids, attention_masks,
                mask_ids, quote_tensor):
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_masks)
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_outputs = []
        for i in range(len(last_hidden_state)):
            hidden_state = last_hidden_state[i]  # [sequence_length, hidden_size]
            mask = hidden_state[mask_ids[i]]
            context = self.dropout(mask)
            context = context.unsqueeze(dim=0)  # context: [1, hidden_size]
            # quote_tensor: [num_class, hidden_size]
            output = torch.mm(context, quote_tensor.t())  # outputs: [1, num_class]
            all_outputs.append(output)
        all_outputs = torch.cat(all_outputs, dim=0)  # all_outputs: [batch, num_class]
        return all_outputs


print("loading model......")
model = QuotRecNet()
model.to(device)


# get rank
def rank_gold(predicts, golds):
    ranks = []
    ps = predicts.data.cpu().numpy()
    gs = golds.cpu().numpy()
    for i in range(len(ps)):
        predict = ps[i]
        gold_index = gs[i]
        predict_value = predict[gold_index]
        predict_sort = sorted(predict, reverse=True)
        predict_index = predict_sort.index(predict_value)
        if predict_index == -1:
            break
        ranks.append(predict_index)
    return ranks


# get NDCG@5
def get_NDCG(ranks):
    total = 0.0
    for r in ranks:
        if r < 5:  # k=5
            total += 1.0 / np.log2(r + 2)
    return total / len(ranks)


# get recall@k
def recall(predicts, golds):
    predicts = predicts.data.cpu().numpy()
    golds = golds.cpu().numpy()
    predicts_index = np.argsort(-predicts, axis=1)
    recall_1, recall_3, recall_5, recall_10, recall_20, recall_30 = 0, 0, 0, 0, 0, 0
    recall_100, recall_300, recall_500, recall_1000 = 0, 0, 0, 0
    for i in range(len(golds)):
        if golds[i] in predicts_index[i][:1000]:
            recall_1000 += 1
            if golds[i] in predicts_index[i][:500]:
                recall_500 += 1
                if golds[i] in predicts_index[i][:300]:
                    recall_300 += 1
                    if golds[i] in predicts_index[i][:100]:
                        recall_100 += 1
                        if golds[i] in predicts_index[i][:30]:
                            recall_30 += 1
                            if golds[i] in predicts_index[i][:20]:
                                recall_20 += 1
                                if golds[i] in predicts_index[i][:10]:
                                    recall_10 += 1
                                    if golds[i] in predicts_index[i][:5]:
                                        recall_5 += 1
                                        if golds[i] in predicts_index[i][:3]:
                                            recall_3 += 1
                                            if golds[i] in predicts_index[
                                                    i][:1]:
                                                recall_1 += 1
    return recall_1, recall_3, recall_5, recall_10, recall_20, recall_30, recall_100, recall_300, recall_500, recall_1000


def training_mask(model, epoch, train, valid, quote_tensor, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(
        total, trainable))
    t_batch = len(train)
    v_batch = len(valid)
    learning_rate = 5e-5
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_MRR = 0
    count = 0
    quote_tensor = quote_tensor.to(device)
    for epoch in range(epoch):
        start = time.clock()
        print("epoch: ", epoch + 1)
        total_loss, total_MRR, total_NDCG = 0, 0, 0
        # train
        for i, (input_ids, token_type_ids, attention_masks, mask_ids, labels) in enumerate(train):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_masks = attention_masks.to(device)
            mask_ids = mask_ids.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(input_ids, token_type_ids, attention_masks, mask_ids, quote_tensor)  # outputs: (batch, num_class)
            # print("outputs:", outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ranks = rank_gold(outputs, labels)
            MRR = np.average([1.0 / (r + 1) for r in ranks])
            NDCG = get_NDCG(ranks)
            total_loss += loss.item()
            total_MRR += MRR
            total_NDCG += NDCG
        end = time.clock()
        print('Epoch running time :{:.0f}'.format(end - start))
        print('Train | Loss:{:.3f} MRR: {:.3f} NDCG: {:.3f}'.format(total_loss, total_MRR/t_batch, total_NDCG/t_batch))

        # validation
        model.eval()
        with torch.no_grad():
            total_loss, total_MRR, total_NDCG = 0, 0, 0
            for i, (input_ids, token_type_ids, attention_masks, mask_ids, labels) in enumerate(valid):
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_masks = attention_masks.to(device)
                mask_ids = mask_ids.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(input_ids, token_type_ids, attention_masks, mask_ids, quote_tensor)
                loss = criterion(outputs, labels)
                ranks = rank_gold(outputs, labels)
                MRR = np.average([1.0 / (r + 1) for r in ranks])
                NDCG = get_NDCG(ranks)
                total_loss += loss.item()
                total_MRR += MRR
                total_NDCG += NDCG
            print("Valid | Loss:{:.5f} MRR: {:.3f} NDCG: {:.3f}".format(
                total_loss, total_MRR / v_batch,
                total_NDCG / v_batch))
        if total_MRR > best_MRR:
            best_MRR = total_MRR
            torch.save(model, "./model/model_ancient_chinese.model")
            print('saving model with MRR {:.3f} NDCG: {:.3f}'.format(
                total_MRR / v_batch, total_NDCG / v_batch))
            count = 0
        else:
            learning_rate = learning_rate * 0.9
            count += 1
        # early stopping
        if count == 3:
            break
        model.train()

      
# Mask Dataset and DataLoader
class Dataset_Mask(Dataset):

    def __init__(self, input_ids, token_type_ids, attention_masks, mask_ids,
                 y):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_masks = attention_masks
        self.mask_ids = mask_ids
        self.label = y

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.label is None:
            return self.input_ids[idx], self.token_type_ids[
                idx], self.attention_masks[idx], self.mask_ids[idx]
        return self.input_ids[idx], self.token_type_ids[
            idx], self.attention_masks[idx], self.mask_ids[idx], self.label[
                idx]


print("loading train and valid dataloader ...")
train_dataset_mask = Dataset_Mask(input_ids=train_input_ids,
                                  token_type_ids=train_token_type_ids,
                                  attention_masks=train_attention_masks,
                                  mask_ids=train_mask_ids,
                                  y=y_train)
train_loader_mask = DataLoader(dataset=train_dataset_mask,
                               batch_size=64,
                               shuffle=True,
                               num_workers=2)
valid_dataset_mask = Dataset_Mask(input_ids=valid_input_ids,
                                  token_type_ids=valid_token_type_ids,
                                  attention_masks=valid_attention_masks,
                                  mask_ids=valid_mask_ids,
                                  y=y_valid)
valid_loader_mask = DataLoader(dataset=valid_dataset_mask,
                               batch_size=64,
                               shuffle=True,
                               num_workers=2)
print("start traing......")
training_mask(model=model,
              epoch=3,
              train=train_loader_mask,
              valid=valid_loader_mask,
              quote_tensor=quote_embeddings,
              device=device)


def test(model, test_loader, quote_tensor, device):
    print("start test......")
    model.eval()
    t_batch = len(test_loader)
    criterion = nn.CrossEntropyLoss()
    quote_tensor = quote_tensor.to(device)
    with torch.no_grad():
        total_loss, total_MRR, total_NDCG, total_ranks = 0, 0, 0, 0
        total_recall_1, total_recall_3, total_recall_5, total_recall_10, total_recall_20, total_recall_30 = 0, 0, 0, 0, 0, 0
        total_recall_100, total_recall_300, total_recall_500, total_recall_1000 = 0, 0, 0, 0
        all_ranks = []
        for i, (input_ids, token_type_ids, attention_masks, mask_ids, labels) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_masks = attention_masks.to(device)
            mask_ids = mask_ids.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(input_ids, token_type_ids, attention_masks, mask_ids, quote_tensor)
            loss = criterion(outputs, labels)
            ranks = rank_gold(outputs, labels)
            all_ranks.extend(ranks)
            MRR = np.average([1.0 / (r + 1) for r in ranks])
            NDCG = get_NDCG(ranks)
            recall_1, recall_3, recall_5, recall_10, recall_20, recall_30, recall_100, recall_300, recall_500, recall_1000 = recall(
                outputs, labels)
            total_loss += loss.item()
            total_MRR += MRR
            total_NDCG += NDCG
            total_ranks += np.sum(ranks)
            total_recall_1 += recall_1
            total_recall_3 += recall_3
            total_recall_5 += recall_5
            total_recall_10 += recall_10
            total_recall_20 += recall_20
            total_recall_30 += recall_30
            total_recall_100 += recall_100
            total_recall_300 += recall_300
            total_recall_500 += recall_500
            total_recall_1000 += recall_1000
        print(len(all_ranks))
        print(all_ranks)
        print(
            "Test | Loss:{:.5f} MRR: {:.3f} NDCG: {:.3f} Mean Rank: {:.0f} Median Rank: {:.0f} Variance: {:.0f}"
            .format(total_loss, total_MRR / t_batch,
                    total_NDCG / t_batch, np.mean(all_ranks),
                    np.median(all_ranks)+1,
                    np.std(all_ranks)))
        print(
            "Recall@1:{:.4f} Recall@3: {:.4f} Recall@5: {:.4f} Recall@10: {:.4f} Recall@20: {:.4f} Recall@30: {:.4f} Recall@100: {:.4f} Recall@300: {:.4f} Recall@500: {:.4f} Recall@1000: {:.4f}"
            .format(
                total_recall_1 / len(y_test), total_recall_3 / len(y_test),
                total_recall_5 / len(y_test), total_recall_10 / len(y_test),
                total_recall_20 / len(y_test), total_recall_30 / len(y_test),
                total_recall_100 / len(y_test), total_recall_300 / len(y_test),
                total_recall_500 / len(y_test),
                total_recall_1000 / len(y_test)))


print("loading test tensor......")
test_input_ids, test_token_type_ids, test_attention_masks, test_mask_ids = make_context_tensors(test_former, test_latter)
test_dataset_mask = Dataset_Mask(input_ids=test_input_ids,
                                 token_type_ids=test_token_type_ids,
                                 attention_masks=test_attention_masks,
                                 mask_ids=test_mask_ids,
                                 y=y_test)
test_loader_mask = DataLoader(dataset=test_dataset_mask,
                              batch_size=64,
                              num_workers=2)
print('loading model ...')
model = torch.load('./model/model_ancient_chinese.model')
model.to(device)
test(model=model,
     test_loader=test_dataset_mask,
     quote_tensor=quote_embeddings,
     device=device)
