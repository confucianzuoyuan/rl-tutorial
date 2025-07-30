```mermaid
graph LR
    A["提示词+人类精修后的回答"] ---> B["DeepSeek-V3基础模型"] ---> C["预测下一个token"] ---> D["比较和目标token的差异，计算损失"] ---> E["更新模型参数"] ---> B
```