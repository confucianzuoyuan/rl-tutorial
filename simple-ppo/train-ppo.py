from typing import Optional
import torch
from common import ModelPPO, device, tokenizer, generate

model_ppo = ModelPPO(torch.load('gen.model', weights_only=False))
model_ppo_ref = ModelPPO(torch.load('gen.model', weights_only=False))

for i in model_ppo_ref.parameters():
    i.requires_grad_(False)

model_cls = torch.load('cls.model', weights_only=False)
model_cls.to(device)

for i in model_cls.parameters():
    i.requires_grad_(False)
    
@torch.no_grad()
def get_question():
    label, question, _ = tokenizer.get_batch_data(prefix=True)
    label = torch.LongTensor(label).to(device)

    #只要问题部分,等号后面的内容切除
    question = [i[:i.index(tokenizer.encoder['=']) + 1] for i in question]

    #统一长度
    lens = max([len(i) for i in question])
    question = [[tokenizer.encoder['P']] * (lens - len(i)) + i
                for i in question]

    question = torch.LongTensor(question).to(device)

    return label, question


label, question = get_question()

print(label, question[:10])

#如果question的长度确定,这里可以转换成批运算
@torch.no_grad()
def get_answer(question):
    answer = generate(model_ppo.model_gen, question)

    #裁剪,只要生成的部分
    answer = answer[:, question.shape[1]:]

    return answer


answer = get_answer(question)

print(answer[:10])

@torch.no_grad()
def get_reward(question, answer, label):
    input_ids = torch.cat((question, answer), 1)
    attention_mask = (input_ids != tokenizer.encoder['P']).long()

    with torch.no_grad():
        logits = model_cls(input_ids=input_ids, attention_mask=attention_mask)

    return logits.gather(1, label.reshape(-1, 1)).squeeze(1)


reward = get_reward(question, answer, label)

print(reward)

"""注释代码,可以不看"""


#get_delta函数的原理解释,注释性代码
#数学上和get_delta函数等价,但是运行效率低
def get_delta_note(value, reward_kl):
    #下一个词的value,减去当前词的value,相当于对value去基线,缩小数值方差
    #每个词的value是相互独立的,前后词value的差,可以视为预测质量的衡量
    value_next = torch.zeros_like(value)
    value_next[:, :-1] = value[:, 1:].clone()

    #在value中融合reward,kl
    diff = reward_kl + value_next - value

    #蒙特卡洛采样法估计Q函数,每个时刻的价值,等于后续所有价值的加权求和
    #这里计算的其实就是adv
    delta = []
    for i in range(diff.shape[1]):
        s = 0
        for j in range(i, diff.shape[1]):
            s += diff[:, j] * 0.95**(j - i)
        delta.append(s)

    return torch.stack(delta, dim=1)


#只用一次循环就计算出delta,计算效率提高很多
def get_delta_fast(value, reward_kl):
    delta = []

    for i in reversed(range(reward_kl.shape[1])):
        value_next = 0
        if i < reward_kl.shape[1] - 1:
            value_next = value[:, i + 1]

        diff = reward_kl[:, i] + value_next - value[:, i]

        diff_last = 0
        if len(delta):
            diff_last = delta[-1]

        delta.append(diff + 0.95 * diff_last)

    return torch.stack(delta[::-1]).transpose(0, 1)


#测试两个函数是等价的,误差是由于计算机精度导致的
for _ in range(200):
    value = torch.randn(64, 26)
    reward_kl = torch.randn(64, 26)

    assert (get_delta_note(value, reward_kl) -
            get_delta_fast(value, reward_kl)).abs().max() < 1e-5

print('success')

import torch.nn.functional as F

def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def logprobs_from_logits(logits, labels):
    # Apply LogSoftmax to get log-probabilities over all classes
    log_probs_all = F.log_softmax(logits, dim=-1)
    
    # Gather the log-probabilities corresponding to the labels
    # labels need to be unsqueezed to match dimensions for gather
    log_probs_selected = torch.gather(log_probs_all, -1, labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_selected

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

class PPOTrainer:

    def __init__(self):
        self.optimizer = torch.optim.Adam(model_ppo.parameters(), lr=1e-5)

    def step(self, question, answer, reward):
        with torch.no_grad():
            #编码
            token = [q.tolist() + a.tolist() for q, a in zip(question, answer)]
            input_ids, attention_mask = tokenizer.batch_pad(token=token)
            del token
            input_ids = torch.LongTensor(input_ids).to(device)
            attention_mask = torch.LongTensor(attention_mask).to(device)

            #question和answer不需要内容,只需要长度信息即可
            lens_q = [question.shape[1]] * len(question)
            lens_a = []

            for a in answer:
                if tokenizer.encoder['E'] in a:
                    lens_a.append(a.tolist().index(tokenizer.encoder['E']) + 1)
                    continue
                lens_a.append(len(a))

            del question
            del answer

            #根据question计算answer的概率,并计算每个动作的分数
            prob_log, value, mask = self.batched_forward_pass(
                model_ppo, input_ids, attention_mask, lens_q, lens_a)

            #使用ref模型计算概率,这是为了计算kl散度
            prob_log_ref, _, _ = self.batched_forward_pass(
                model_ppo_ref, input_ids, attention_mask, lens_q, lens_a)

            #计算两份概率的kl散度,并融入reward
            reward = self.compute_rewards(reward, prob_log, prob_log_ref, mask)

            #计算delta和target,用于计算loss
            value, delta, target = self.compute_advantages(value, reward, mask)

        #每批数据循环N次模型
        for _ in range(4):
            #每次算一个数据
            for i in range(len(input_ids)):
                #重新计算概率和value
                prob_log_new, value_new, _ = self.batched_forward_pass(
                    model_ppo, input_ids[i].unsqueeze(0),
                    attention_mask[i].unsqueeze(0), [lens_q[i]], [lens_a[i]])

                #根据新旧概率求出变化率,进而求出loss
                #根据target和value的差可以计算出另外一份loss
                loss = self.get_loss(prob_log[i].unsqueeze(0),
                                     value[i].unsqueeze(0), prob_log_new,
                                     value_new, mask[i].unsqueeze(0),
                                     delta[i].unsqueeze(0),
                                     target[i].unsqueeze(0))

                if not loss:
                    continue

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model_ppo.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

    def batched_forward_pass(self, model, input_ids, attention_mask, lens_q,
                             lens_a):
        logits, value = model(input_ids=input_ids,
                              attention_mask=attention_mask)

        #取每个字的概率对数
        prob_log = logprobs_from_logits(logits[:, :-1], input_ids[:, 1:])

        #是预测结果并且不是PAD的位置是1
        mask = torch.zeros_like(attention_mask)
        mask[:, :-1] = attention_mask[:, 1:]
        for i in range(len(input_ids)):
            start = lens_q[i] - 1
            end = start + lens_a[i]
            mask[i, :start] = 0
            mask[i, end:] = 0

        #对最后一个字的预测没有意义,直接丢弃
        value = value[:, :-1]
        mask = mask[:, :-1]

        return prob_log, value, mask

    def compute_rewards(self, reward, prob_log, prob_log_ref, mask):
        reward_kl = []

        for i in range(len(reward)):
            #求两份概率的kl散度
            kl = self.get_kl(prob_log[i], prob_log_ref[i]) * -0.2

            #把reward加在最后一个字的kl散度上
            if (mask[i] == 0).all():
                #print('all 0')
                idx = 0
            else:
                idx = mask[i].nonzero()[-1].item()
            kl[idx] += reward[i]

            reward_kl.append(kl)

        return torch.stack(reward_kl)

    def compute_advantages(self, value, reward_kl, mask):
        value = value * mask
        reward_kl = reward_kl * mask

        #这里计算delta的过程,可以看上面的注释.
        delta = []
        for i in reversed(range(reward_kl.shape[1])):
            value_next = 0
            if i < reward_kl.shape[1] - 1:
                value_next = value[:, i + 1]

            diff = reward_kl[:, i] + value_next - value[:, i]

            diff_last = 0
            if len(delta):
                diff_last = delta[-1]

            delta.append(diff + 0.95 * diff_last)

        delta = torch.stack(delta[::-1]).transpose(0, 1)

        #定义target,它估计了理想的value值
        target = delta + value
        delta = masked_whiten(delta, mask)

        return value, delta, target

    def get_loss(self, prob_log, value, prob_log_new, value_new, mask, delta,
                 target):

        #对数概率,相除变相减,取exp后还原为商,即两个模型输出logits的变化率
        ratio = (prob_log_new - prob_log).exp()

        #如果变化率太过于剧烈,可能是发生了震荡,跳过
        if masked_mean(ratio, mask).item() > 10:
            #print('skip', masked_mean(ratio, mask).item())
            return None

        #先算两个value的loss,简单的算mse loss就可以了
        loss_vf1 = (value_new - target)**2
        #数值裁剪,很显然是为了缓解自举
        loss_vf2 = torch.clip(value_new, value - 0.2, value + 0.2)
        loss_vf2 = (loss_vf2 - target)**2
        #两份loss取大的,还是为了缓解自举
        loss_vf = 0.5 * masked_mean(torch.max(loss_vf1, loss_vf2), mask)

        #计算ppo loss
        loss_surr1 = -delta * ratio
        #数值裁剪,很显然是为了缓解自举
        loss_surr2 = -delta * ratio.clamp(0.8, 1.2)
        loss_surr = masked_mean(torch.max(loss_surr1, loss_surr2), mask)

        return loss_surr + 0.1 * loss_vf

    def get_kl(self, a, b):
        method = 'kl'

        if method == 'kl':
            return a - b

        if method == 'abs':
            return (a - b).abs()

        if method == 'mse':
            return (a - b).square() * 0.5

        if method == 'full':
            return torch.nn.functional.kl_div(a,
                                              b,
                                              log_target=True,
                                              reduction='none')


trainer = PPOTrainer()

trainer.step(question, answer, reward)

for epoch in range(2000):
    label, question = get_question()
    answer = get_answer(question)
    reward = get_reward(question, answer, label)

    trainer.step(question, answer, reward)

    if epoch % 100 == 0:
        print(epoch, reward.mean().item())
        for _, q, a, r in zip(range(2), question, answer, reward):
            q = tokenizer.decode(q.tolist())
            a = tokenizer.decode(a.tolist())
            r = r.item()
            print(q, a, r)

model_ppo.to('cpu')
torch.save(model_ppo, 'ppo.model')



