import random


class Tokenizer:

    def __init__(self):
        self.vocab = {
            'mark': list('PSEU'),
            'number': list('0123456789'),
            'letter': list('pqwertyuio'),
            'chinese_lower': list('〇一二三四五六七八九'),
            'chinese_upper': list('零壹贰叁肆伍陆柒捌玖'),
            'other': list('数字大写小母:=_'),
        }

        self.decoder = [j for i in self.vocab.values() for j in i]
        self.encoder = {j: i for i, j in enumerate(self.decoder)}

        self.label = {
            'number': 0,
            'letter': 1,
            'chinese_lower': 2,
            'chinese_upper': 3
        }
        self.prefix = ['数字', '字母', '小写', '大写']

    def decode(self, x):
        return ''.join([self.decoder[i] for i in x])

    def get_data(self, prefix):
        #生成问题和答案
        question = random.randint(1000, 9999)
        answer = int(str(question) * 4) * 4
        #answer = question**8
        
        question = list(str(question))
        answer = list(str(answer))

        #随机label
        label = random.choice(list(self.label.keys()))

        #根据label替换答案成其他字符集
        answer = [self.vocab[label][int(i)] for i in answer]

        #label转数字
        label = self.label[label]

        #组合问题和答案
        if prefix:
            prefix = list(self.prefix[label])
        else:
            prefix = list('__')
        token = prefix + [':'] + question + ['='] + answer

        #编码
        token = [self.encoder[i] for i in token]
        token = [self.encoder['S']] + token + [self.encoder['E']]

        return label, token

    def get_batch_data(self, prefix):
        data = [self.get_data(prefix=prefix) for _ in range(64)]

        label = [i[0] for i in data]
        token = [i[1] for i in data]

        return label, *self.batch_pad(token=token)

    def batch_pad(self, text=None, token=None):
        if text:
            #编码
            token = [[self.encoder[j] for j in i] for i in text]

        lens = max([len(i) for i in token])

        input_ids = []
        attention_mask = []
        for i in token:
            attention_mask.append([1] * len(i) + [0] * (lens - len(i)))
            input_ids.append(i + [self.encoder['P']] * (lens - len(i)))

        return input_ids, attention_mask


tokenizer = Tokenizer()

r = [tokenizer.decode(i) for i in tokenizer.get_batch_data(prefix=True)[1]][:10]
print(r)

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

class ModelGEN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        from transformers import GPT2Config, GPT2Model

        self.config = GPT2Config(bos_token_id=tokenizer.encoder['S'],
                                 eos_token_id=tokenizer.encoder['E'],
                                 n_embd=64,
                                 n_head=4,
                                 n_layer=4,
                                 n_positions=128,
                                 vocab_size=len(tokenizer.decoder))

        self.feature = GPT2Model(self.config)

        self.fc_out = torch.nn.Linear(64, self.config.vocab_size, bias=False)

        self.to(device)
        self.train()

    def forward(self, input_ids, attention_mask):
        out = self.feature(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state

        return self.fc_out(out)
    
class ModelCLS(torch.nn.Module):

    def __init__(self):
        super().__init__()
        from transformers import BertConfig, BertModel

        self.config = BertConfig(hidden_size=64,
                                 intermediate_size=64,
                                 max_position_embeddings=128,
                                 num_attention_heads=4,
                                 num_hidden_layers=4,
                                 vocab_size=len(tokenizer.decoder))

        self.feature = BertModel(self.config)

        self.fc_out = torch.nn.Sequential(torch.nn.Dropout(p=0.1),
                                          torch.nn.Linear(64, 4))

        self.to(device)
        self.train()

    def forward(self, input_ids, attention_mask):
        out = self.feature(input_ids=input_ids,
                           attention_mask=attention_mask).pooler_output

        return self.fc_out(out)
    
class ModelPPO(torch.nn.Module):

    def __init__(self, model_gen):
        super().__init__()
        self.model_gen = model_gen
        self.v_head = torch.nn.Sequential(torch.nn.Dropout(0.1),
                                          torch.nn.Linear(64, 1))

        self.to(device)
        self.train()

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model_gen.feature(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True).last_hidden_state

        logits = self.model_gen.fc_out(last_hidden_state)
        value = self.v_head(last_hidden_state).squeeze(-1)

        return logits, value
    
generater = None


def generate(model_gen, input_ids):
    global generater
    if not generater:
        #包装类,用于生成
        from transformers import GPT2LMHeadModel
        generater = GPT2LMHeadModel(model_gen.config)
        generater.transformer = model_gen.feature
        generater.lm_head = model_gen.fc_out
        generater.to(device)

    return generater.generate(input_ids=input_ids,
                              min_length=-1,
                              top_k=0.0,
                              top_p=1.0,
                              do_sample=True,
                              pad_token_id=tokenizer.encoder['P'],
                              max_new_tokens=25,
                              eos_token_id=tokenizer.encoder['E'])