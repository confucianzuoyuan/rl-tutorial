import torch
from common import tokenizer, device, generate

def get_batch_data():

    def pad(data, split, lens):
        #做个白板
        input_ids = torch.full((len(data), lens),
                               tokenizer.encoder['P'],
                               device=device)

        #往白板里黏贴数据
        for i, d in enumerate(data):
            input_ids[i, :len(d)] = torch.LongTensor(d)

        attention_mask = (input_ids != tokenizer.encoder['P']).long()

        #计算label
        label = input_ids.clone()
        for l, s in zip(label, split):
            #问题和pad的位置是-100
            l[:s] = -100
            l[l == tokenizer.encoder['P']] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

    #正确的问答
    choice = [tokenizer.get_data(third_number=True) for i in range(64)]

    #错误的回答简单地定义为空回答就可以了
    split = [i.index(tokenizer.encoder['=']) + 1 for i in choice]
    reject = [d[:s] for d, s in zip(choice, split)]
    reject = [i + [tokenizer.encoder['E']] for i in reject]

    #求最大长度
    lens = max([len(i) for i in choice])

    return pad(choice, split, lens), pad(reject, split, lens)


print(get_batch_data())

model_dpo = torch.load('gen.model', weights_only=False)
model_dpo.to(device)
model_dpo.train()

model_dpo_ref = torch.load('gen.model', weights_only=False)
model_dpo_ref.to(device)
model_dpo_ref.train()

def get_prob_log(model, choice, reject):
    b = choice['input_ids'].shape[0]

    #合并两部分输入,同时计算以提高效率
    #[b, 21]
    input_ids = torch.cat([choice['input_ids'], reject['input_ids']], dim=0)
    attention_mask = torch.cat(
        [choice['attention_mask'], reject['attention_mask']], dim=0)
    label = torch.cat([choice['label'], reject['label']], dim=0)

    #[b, 21, 28]
    out = model(input_ids=input_ids, attention_mask=attention_mask)

    #偏移以对齐
    #[b, 20]
    label = label[:, 1:]
    #[b, 20, 28]
    out = out[:, :-1]

    #取所有字的预测概率,因为要求联合概率,所以取对数
    out = (out.softmax(2) + 1e-8).log()

    #取预测到label的概率
    #索引不能是负数,所以这里把负数置0
    index = label.clone().unsqueeze(2)
    index[index == -100] = 0
    prob = out.gather(2, index=index).squeeze(2)

    #只取答案部分的loss,筛选后,所有答案的概率对数求和
    prob = (prob * (label != -100)).sum(1)

    #choice和reject的预测概率求差
    return prob[:b] - prob[b:]


print(get_prob_log(model_dpo, *get_batch_data()))

optimizer = torch.optim.Adam(model_dpo.parameters(),
                             lr=1e-4,
                             betas=(0.9, 0.999),
                             eps=1e-8)

for i in range(20_0000):
    choice, reject = get_batch_data()

    #两个模型分别计算概率对数
    prob_log = get_prob_log(model_dpo, choice, reject)
    with torch.no_grad():
        prob_log_ref = get_prob_log(model_dpo_ref, choice, reject)

    #两份概率计算kl散度
    kl = -0.1 * (prob_log - prob_log_ref)

    #以kl散度计算loss
    loss = (kl.sigmoid() + 1e-8).log().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 2000 == 0:
        question = tokenizer.get_data(third_number=True)
        question = question[:question.index(tokenizer.encoder['=']) + 1]
        question = torch.LongTensor(question).unsqueeze(0).to(device)

        gen = generate(model_dpo, question)
        print(i, tokenizer.decode(gen[0].tolist()))

model_dpo.to('cpu')
torch.save(model_dpo, 'dpo.model')



