import torch
from common import tokenizer, device, ModelGEN, generate

def get_batch_data():
    data = [tokenizer.get_data(third_number=False) for i in range(64)]

    #求最大长度
    lens = max([len(i) for i in data])

    #做个白板
    input_ids = torch.full((len(data), lens),
                           tokenizer.encoder['P'],
                           device=device)

    #往白板里黏贴数据
    for i, d in enumerate(data):
        input_ids[i, :len(d)] = torch.LongTensor(d)

    attention_mask = (input_ids != tokenizer.encoder['P']).long()

    return input_ids, attention_mask


print(get_batch_data())

model_gen = ModelGEN()
optimizer = torch.optim.Adam(model_gen.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.encoder['P'])

for i in range(1000):
    input_ids, attention_mask = get_batch_data()

    out = model_gen(input_ids=input_ids, attention_mask=attention_mask)

    loss = criterion(out[:, :-1].flatten(end_dim=1), input_ids[:,
                                                               1:].flatten())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
        gen = generate(model_gen, input_ids[0].unsqueeze(0))
        print(i, tokenizer.decode(gen[0].tolist()))

model_gen.to('cpu')
torch.save(model_gen, 'gen.model')