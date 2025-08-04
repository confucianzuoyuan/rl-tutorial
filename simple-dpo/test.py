import torch
from common import device, tokenizer, generate

model_dpo = torch.load('dpo.model', weights_only=False)
model_dpo.to(device)
model_dpo.eval()

#随机一批数据
input_ids = [tokenizer.get_data(third_number=True) for i in range(64)]

#切分成question和answer
split = [i.index(tokenizer.encoder['=']) + 1 for i in input_ids]
question = [input_ids[i][:split[i]] for i in range(len(input_ids))]
answer = [input_ids[i][split[i]:] for i in range(len(input_ids))]

#根据question生成predict
input_ids = [torch.LongTensor(i).unsqueeze(0).to(device) for i in question]
predict = [generate(model_dpo, i) for i in input_ids]

#裁剪,只要生成的部分
predict = [p[0].tolist()[len(q):] for p, q in zip(predict, question)]

#解码成文本
question = [tokenizer.decode(i) for i in question]
answer = [tokenizer.decode(i) for i in answer]
predict = [tokenizer.decode(i) for i in predict]

for q, a, p in zip(question, answer, predict):
    try:
        diff = abs(float(q[1:-1]) - eval(p[:p.index('E')]))
    except:
        diff = abs(float(q[1:-1]))

    print(q, p, diff)
    
