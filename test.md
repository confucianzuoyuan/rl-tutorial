> [!NOTE]
> PPO, 演员-评论家风格的实现
> 1. **for** $i=1,2,...$ **do**
> 2. > **for** $\text{actor}=1,2,...,N$ **do**
> 3. >> 在环境中将旧的策略 $\pi_{\theta_{old}}$ 运行 $T$ 个时间步
> 4. >> 计算优势 $\hat{A}_1,...,\hat{A}_T$
> 5. > **end for**
> 6. > 针对参数 $\theta$ 优化裁剪目标函数 $L$ ，运行 ==$K$ 个 Epoch== 。批次大小为 $M \le NT$ 。
> 7. > $\theta_{old} \leftarrow \theta$
> 8. **end for**