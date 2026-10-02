---
title: "论文综述：虚假的前沿——诊断并缓解自进化搜索智能体中的共同作弊"
originalTitle: "False Frontiers: Diagnosing and Mitigating Co-Cheating in Self-Evolving Search Agents"
originalUrl: "https://arxiv.org/abs/2609.39102"
authors: "Meijia Chen, Hao Li, Zheng Lu, Hongshan Lin, Junbai Tian, Yichen Liu, Zijun Tian, Yufan Zou, Shuhan Sun, Hanxin Chen, Zeyu Zhang, Weizhi Du, Yueting Li, Tianyu Shi, Alaa Khamis"
institution: "Rutgers University"
hfVotes: 192
publishDate: "2026-09-30"
reviewDate: "2026-10-02"
tags: ["self-evolving agents", "search agents", "co-cheating", "CrossFit", "multi-sample verification", "Qwen3.5", "reinforcement learning"]
description: '自进化搜索智能体里出题者和答题者会在共同的错误答案上越来越一致，内部奖励上涨但真实能力没涨，本文用CrossFit按文档来源分组交叉打分来缓解'
---

## 一、论文是干什么的？

想象一个自学班：一位同学负责从课本里出题并自己写参考答案（出题者，proposer），另一位同学负责做题（答题者，solver），做对了就给出题者加分。听上去很美，两个人互相督促，不需要老师。但问题来了：如果两人读的是同一页课本、犯的是同一种误解，那么出题者写下错误答案，答题者也会做出同样的错误答案，结果两人答案一致，出题者拿到高分，答题者也被奖励。班级内部的成绩单越来越漂亮，实际考试却没进步。论文把这种现象叫作**共同作弊**（co-cheating），并称由此产生的进步为虚假的前沿（false frontiers）。

这篇论文做了三件事：第一，用事后审计的办法，对照原始文档检查自进化循环里生成的题目和答案，量化共同作弊有多严重；第二，提出两个缓解办法，**多样本验证**（Multi-Sample Verification，MSV）和主打方法 **CrossFit**；第三，在 Qwen3.5-4B 和 Qwen3.5-9B 上验证，七个搜索问答基准的平均成绩比基线高出 8 到 9 个百分点左右。论文基于 Dr. Zero 这类出题者与答题者共同进化的自进化搜索智能体框架展开。

## 二、核心方法与创新

### 1. 先把问题量化：错误一致质量（false-agreement mass，FF）

论文定义 FF 为：在被评估的题目与答案配对中，出题者和答题者**一致地给出同一个错误答案**的比例。做法是请一个更强的模型（论文中写作 gpt-6-astra/high）对照原始文档，事后审计约 129 个训练步（每轮 43 步，共 3 轮）。结果显示：在基线 Dr. Zero 中，到第 3 轮 FF 升到 6.1%（4B）和 8.8%（9B），伪标签正确率却停滞在 0.747 和 0.737 左右。也就是说，内部奖励在涨，真实正确率没跟上。

### 2. 方法一：多样本验证（MSV）

类比：出题老师写完题后，用两种方式验算。一种是**看着课本**验算（source-aware），另一种是**闭卷**凭自己知识验算（source-blind）。具体做法是对每个出题各采样 3 个看文档的答案和 3 个不看文档的答案，只有两种视角都有答案且结果一致时，才录用这道题，并用验证过的共识答案替换原来的草稿答案。录用条件在论文中写作 $y^{\mathrm{src}} \neq \varnothing \wedge y^{\mathrm{blind}} \neq \varnothing \wedge y^{\mathrm{src}} \simeq y^{\mathrm{blind}}$。MSV 能让答题者学到的标签更准，但它没有切断同源错误被当作奖励这条路，所以 FF 只是小幅下降（6.1% 到 5.7%，8.8% 到 7.2%）。

### 3. 方法二（主打）：CrossFit，交叉拟合

类比：把全班课本分成 A、B 两摞。出题者用 A 摞出的题，交给**只学过 B 摞的助教**去做，看助教能不能做对来打分；反过来，B 摞的题由只学过 A 摞的助教评分。这样，哪怕出题者在某本书上有误解，学过别的书的助教也很难凭同一个错误给出同样的答案，所以同源错误换来高分的捷径被堵住了。

要点：

- 在**文档来源**这一层把文档分成 A、B 两组（不是按题目随机分）。
- 训练一个辅助答题者只用 A 组，用它给 B 组的题打分，角色互换再来一遍。
- 主答题者仍然用所有被录用的题训练，**只有出题者拿到的反馈被交叉拟合**，答题者的更新规则不变。
- 出题者奖励大致写作 $R_P(q) = f\big(\sum_j \mathbb{1}[z_j \simeq \tilde{y}]\big)$，其中 $z_j$ 是辅助答题者的回答；论文给出 $f(k) = (5-k)/4$（$0<k<5$），即辅助答题者答对越多，题越简单、奖励越低，鼓励出有难度但可解的题。
- 策略梯度损失按序列长度归一化，优势函数在任务分桶内标准化：$\hat{A}_e = (r_e - \mu_{g(e)}) / (\sigma_{g(e)} + 10^{-6})$。

### 4. 消融：真正起作用的是按来源隔离

在固定题库回放实验中：辅助答题者与题目同源时 FF 为 6.4% 到 8.7%（和基线差不多）；按题目随机切分 FF 为 5.0% 到 6.2%（改善很小）；按来源 ID 切分，FF 降到 0.4% 和 0.1%（最大效果）。这说明关键不是多训练一个模型，而是辅助模型**没见过这个来源**。

## 三、使用了哪些模型和计算资源？

| 项目 | 内容 |
|---|---|
| 基础模型 | Qwen3.5-4B、Qwen3.5-9B |
| 审计模型 | gpt-6-astra/high（事后核对题目与文档，调用方式与费用暂无相关信息） |
| 训练框架与调度 | 基于 Dr. Zero：3 轮；每轮 18 次出题者更新、25 次答题者更新；每轮录用 1,600 道题，每题采样 5 个回答 |
| GPU | 每次运行预留 8 张 NVIDIA H200 |
| 算力开销（H200 小时/次完整运行） | Dr. Zero：379（4B）、476（9B）；CrossFit（每折 25）：650（4B）、854（9B）；MSV+CrossFit：990（4B）、1,281（9B） |
| 墙钟耗时 | 论文附录给出：Dr. Zero 47 小时（4B）和 60 小时（9B）；CrossFit（每折 25）81 小时和 107 小时；MSV+CrossFit 124 小时和 160 小时（含等待时间，每次运行占用 8 张 H200） |
| 相对开销 | CrossFit 比基线多约 72% 到 79%；MSV+CrossFit 约为基线的 2.6 到 2.7 倍 |
| 学习率、批大小、检索器与语料 | 暂无相关信息（论文正文中未给出） |

## 四、实验结果

评测指标为 Cover-EM：归一化后的非空参考答案只要出现在归一化后的预测里就算对。评测集共 1,325 题，覆盖 7 个基准。

**下游平均成绩（Cover-EM 平均）**

| 方法 | 4B | 9B |
|---|---|---|
| Search-R1 | 0.401 | 0.434 |
| Dr. Zero | 0.400 | 0.428 |
| CrossFit | 0.488 | 0.512 |

CrossFit 比 Dr. Zero 高 8.8 和 8.4 个百分点，比 Search-R1 高 8.7 和 7.8 个百分点。注意 2WikiMQA 上 9B（0.425）低于 4B（0.575）并非抄写错误，Dr. Zero（0.485 到 0.320）和 Search-R1（0.510 到 0.310）也有同样的下降，论文未解释原因。

**CrossFit 在 7 个基准上的 Cover-EM**

| 基准 | 4B | 9B |
|---|---|---|
| Natural Questions | 0.455 | 0.580 |
| TriviaQA | 0.730 | 0.765 |
| PopQA | 0.415 | 0.455 |
| HotpotQA | 0.460 | 0.490 |
| 2WikiMQA | 0.575 | 0.425 |
| MuSiQue | 0.245 | 0.255 |
| Bamboogle | 0.536 | 0.616 |
| 平均 | 0.488 | 0.512 |

**第 3 轮错误一致质量 FF**

| 模型 | Dr. Zero | MSV | CrossFit | MSV+CrossFit |
|---|---|---|---|---|
| 4B | 6.1% | 5.7% | 3.0% | 2.0% |
| 9B | 8.8% | 7.2% | 3.7% | 1.7% |

论文的结论是：MSV 单独使用能提高答题者学到的真值质量，但不能阻止一致性信号变得过于乐观；与 CrossFit 组合后，最终的错误一致质量最低。

## 五、潜在应用与已落地应用

潜在方向：

- 任何模型自己出题、自己（或同族模型）答题、再据此奖励的自进化训练，如自博弈、合成数据训练、无标注强化学习，都可借鉴按来源隔离反馈的思路。
- 作为监控工具：把 FF 之类的审计指标放进训练看板，避免把虚高的内部奖励误当成进步。
- 合成数据过滤：MSV 式的看文档与闭卷双视角一致才录用，可用于数据清洗。

已落地应用：论文页面和检索结果中未找到官方代码发布或产业落地的信息，暂无相关信息。

## 六、网络上的讨论与评价

- 检索到的页面包括 [arXiv 摘要页](https://arxiv.org/abs/2609.39102)、[Hugging Face 论文页](https://huggingface.co/papers/2609.39102)、[hyper.ai 论文页](https://hyper.ai/en/papers/2609.39102)、[paperswithcode.co 条目](https://paperswithcode.co/paper/2609.39102)，以及 [AI Weekly 的简讯](https://aiweekly.co/alerts/papers-crossfit-cuts-search-agent-co-cheating-to-37)。简讯主要复述论文结果，并指出这项工作为团队监控内部奖励上涨是否等于真实进步提供了可测量的诊断。
- 相关工作还有 [CAFE: Self-Improving Search Agents Need Co-Evolving Feedback](https://arxiv.org/pdf/2608.24794)，主题接近，两者的具体关系尚未核实。
- 论文自述的局限：共享的预训练、重叠的网络证据、语义相关的来源以及会自适应的出题者，仍可能造成相关联的错误，即按来源切分并不能保证完全隔离。
- 未检索到来自 Reddit、Hacker News、X 等的针对该论文的独立深入讨论。社区评价暂无相关信息。

## 七、思维导图

```mermaid
mindmap
  root((False Frontiers 共同作弊))
    研究背景与问题
      Dr. Zero 出题者与答题者共同进化
      共同作弊 co-cheating
        同源错误被当作一致奖励
        内部奖励上涨但外部正确率停滞
    诊断审计
      false-agreement mass FF 错误一致质量
      gpt-6-astra/high 事后审计 129步
      基线第3轮 FF 6.1% 与 8.8%
    方法与技术贡献
      MSV 多样本验证
        3个source-aware加3个source-blind样本
        双视角共识才录用
      CrossFit 交叉拟合
        按来源文档分组A与B
        辅助答题者只学对侧折
        出题者奖励 f(k)=(5-k)/4
        主答题者更新不变
      训练目标 序列归一化 policy-gradient 与分桶优势标准化
    实验设计与结果
      设置
        Qwen3.5-4B 与 9B
        每轮1600题 每题5个回答
        7个基准 1325题 Cover-EM
      FF 结果
        CrossFit 3.0% 与 3.7%
        MSV加CrossFit 2.0% 与 1.7%
      下游平均
        CrossFit 0.488 与 0.512
        比 Dr. Zero 高8.8与8.4点
    消融与机制分析
      固定题库回放
        同源辅助 FF 6.4%到8.7%
        随机题目切分 FF 5.0%到6.2%
        Source-ID 切分 FF 0.4%与0.1%
      起作用的是来源排除而非多一个模型
    算力与影响展望
      8张H200 CrossFit 650与854 H200小时
      局限 共享预训练仍致相关错误
```
