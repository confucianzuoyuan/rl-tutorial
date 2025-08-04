import torch
from common import ModelGEN, tokenizer, device, generate

model_gen = ModelGEN()

optimizer = torch.optim.AdamW(model_gen.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.encoder['P'])

for epoch in range(15000):
    _, input_ids, attention_mask = tokenizer.get_batch_data(prefix=False)
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)

    logits = model_gen(input_ids=input_ids, attention_mask=attention_mask)

    loss = criterion(logits[:, :-1].flatten(end_dim=1),
                     input_ids[:, 1:].flatten())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        print(epoch)
        for i in generate(model_gen, input_ids[:2, :9]):
            print(tokenizer.decode(i.tolist()))

model_gen.to('cpu')
torch.save(model_gen, 'gen.model')

