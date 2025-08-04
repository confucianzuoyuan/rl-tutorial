import torch
from common import tokenizer, device, generate

@torch.no_grad()
def get_question_and_answer():
    _, token, _ = tokenizer.get_batch_data(prefix=True)

    split = [i.index(tokenizer.encoder['=']) + 1 for i in token]

    #只要问题部分,等号后面的内容切除
    question = [t[:s] for t, s in zip(token, split)]
    answer = [t[s:] for t, s in zip(token, split)]

    #统一长度
    lens = max([len(i) for i in question])
    question = [[tokenizer.encoder['P']] * (lens - len(i)) + i
                for i in question]
    question = torch.LongTensor(question).to(device)

    lens = max([len(i) for i in answer])
    answer = [[tokenizer.encoder['P']] * (lens - len(i)) + i for i in answer]
    answer = torch.LongTensor(answer).to(device)

    return question, answer


question, answer = get_question_and_answer()

print(question.shape, answer.shape)

model_ppo = torch.load('ppo.model', weights_only=False)
model_ppo.to(device)
model_ppo.eval()

predict = generate(model_ppo.model_gen, question)
predict = predict[:, question.shape[1]:]

print(predict.shape)

correct = 0
for q, a, p in zip(question, answer, predict):
    q, a, p = q.tolist(), a.tolist(), p.tolist()

    if tokenizer.encoder['E'] in a:
        split = a.index(tokenizer.encoder['E']) + 1
        a = a[:split]

    if tokenizer.encoder['E'] in p:
        split = p.index(tokenizer.encoder['E']) + 1
        p = p[:split]

    q, a, p = tokenizer.decode(q), tokenizer.decode(a), tokenizer.decode(p)

    print(q, a, p)

    correct += a == p

print(correct / len(answer))



