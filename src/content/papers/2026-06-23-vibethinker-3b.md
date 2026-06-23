---
title: "论文综述：VibeThinker-3B 用 3B 小模型逼近顶级推理水平"
originalTitle: "VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models"
originalUrl: "https://arxiv.org/abs/2606.16140"
authors: "Sina Weibo Inc. (WeiboAI) team"
institution: "Sina Weibo Inc. (新浪微博)"
hfVotes: 0
publishDate: "2026-06-15"
reviewDate: "2026-06-23"
tags: ["small-language-model", "reasoning", "reinforcement-learning", "post-training", "test-time-scaling"]
description: '一个仅3B参数的小模型，靠精心设计的后训练流程，在数学和编程等可验证推理任务上逼近甚至追平几百倍大的顶级模型'
---

## 一、论文是干什么的？

如今想让 AI「会推理」（解数学竞赛题、写竞赛级代码），主流做法是把模型越堆越大——几百亿、上千亿参数。大家普遍相信：参数太少（比如 3B 以下）的小模型，注定搞不定复杂的数学推导和编程。

这篇来自**新浪微博 AI 团队**的技术报告偏要挑战这个常识。他们做出了 **VibeThinker-3B**：一个只有 30 亿参数的「小不点」模型。在数学、编程这类**答案能被严格验证对错**的任务上，它的成绩竟然摸到了 DeepSeek V3.2（6710 亿参数）、Kimi K2.5（1 万亿参数）、Gemini 3 Pro 这些「巨无霸」的水平。

打个比方：这就像一个体重只有别人零头的轻量级选手，在解题擂台上和重量级冠军打成平手。论文想证明的核心观点是——**「会不会推理」未必由「个头大小」决定**。这是他们前作 VibeThinker-1.5B 的延续：1.5B 版证明了「小模型也能完整地推理」，而 3B 版进一步追问「到底需要多少参数才能进入顶级推理梯队」。

## 二、核心方法与创新

整个模型基于阿里的 **Qwen2.5-Coder-3B** 基座做「后训练」（post-training，即在预训练好的底座上做专项特训）。整套方法的灵魂叫**光谱到信号原则**（Spectrum-to-Signal Principle, SSP），可以用一句话概括：**先把解题思路铺得又广又散（光谱），再把其中对的思路放大、强化（信号）。**

具体分成环环相扣的几个阶段：

### 1. 监督微调 SFT——先「铺光谱」

这一步教模型尽可能多地学会**各种不同的解题路子**，而不是死记一种标准答案。两个关键设计：

- **多路径蒸馏 + 多样性探索蒸馏**（Diversity-Exploring Distillation）：用强大的「老师模型」对每道题采样多条不同的推理轨迹，全部保留下来，让学生学到多种分解、推导、验证方法。还会把不同领域（数学/代码/STEM）训练出的「专家模型」在参数层面合并，既保住各领域能力又保持多样性。
- **两阶段课程学习**（curriculum）：第一阶段用全量数据广撒网（5 轮训练），第二阶段只挑「难题、长推理」的样本深耕（再训 2 轮）。难题怎么筛？先扔掉推理少于 5K token 的，再用前代 1.5B 模型对每题试做 8 次，把错误率低于 0.75 的「太简单」的题剔除。
- **多级质量管控**：用 N-gram 过滤防止测试集污染、用 LLM 筛掉劣质题目、用代码沙箱执行和多数投票验证答案正确性。

### 2. 强化学习 RL——再「放信号」

核心算法叫 **MGPO**（最大熵引导的策略优化）。它的聪明之处在于**专挑模型「会一半、不会一半」的题来练**——太难的题（几乎全错）和太简单的题（几乎全对）都被降权，把精力集中在模型能力边界上。其权重公式为：

$$w(q)=\exp\left(-\gamma D_{\mathrm{ME}}(p(q)\|p_0)\right),\quad p_0=0.5$$

其中 $p(q)$ 是模型对某题的正确率，越接近 0.5（最不确定）权重越高。就像老师辅导学生，不会浪费时间在「肯定会」和「怎么都不会」的题上，而是死磕「再推一把就能突破」的题。

RL 按 **数学 → 代码 → STEM** 顺序依次进行，用单一的 **64K 超长上下文**窗口（避免截断破坏长推理）。每个领域用不同的「验证器」给奖励：数学查最终答案、代码跑沙箱测试、STEM 查选项匹配。

此外还有 **Long2Short 数学 RL**：在保证正确率不掉的前提下，奖励更短的正确答案、惩罚啰嗦的正确答案，让模型「想得对又说得简洁」。这是个零和的奖励重分配，不会扰乱整体基线。

### 3. 离线自蒸馏 + Instruct RL——「收口」

- **离线自蒸馏**（Offline Self-Distillation）：把数学、代码、STEM 三个 RL 阶段产出的高质量推理轨迹收集起来，「回灌」给一个统一的学生模型，把分散的能力融合成一体。还引入「学习潜力分数」——专挑那些老师做对了、但学生还没学会的轨迹来蒸馏，把好钢用在刀刃上。
- **Instruct RL**：最后再做一轮指令对齐，让模型在变强推理的同时，依然能严格听话（遵守格式、字数、关键词等复杂约束）。

### 4. 测试时增强 CLR——答题时再加一层保险

**断言级可靠性评估**（Claim-Level Reliability Assessment, CLR）是一种「测试时扩展」技巧：答题时模型先生成 $K=32$ 条候选解答，每条提取 $M=5$ 个关键论断，然后模型**自己当裁判**去逐条证伪/验证这些论断，算出每条轨迹的可靠性分数：

$$r_{k}=\left(\frac{1}{M}\sum_{m=1}^{M}v_{k,m}\right)^{M}$$

这个 $M$ 次方设计会**狠狠惩罚有逻辑漏洞的解答**（只要有一个关键论断错，分数就大幅下降）。最后按可靠性加权投票选出最终答案。相比对整条长轨迹做验证，CLR 只盯关键「逻辑锚点」，省 token 还更准。

### 理论升华

作者由此提出两个观点：**参数压缩-覆盖假说**（可验证推理可以被压缩进一个紧凑的「推理内核」，而开放领域知识则需要海量参数去「覆盖」事实）；以及**推理-知识解耦范式**（大模型擅长存广博知识，小模型足以承载高密度推理深度）。这解释了为何 VibeThinker-3B 在数学/代码上能追平大模型，却在知识密集的 GPQA-Diamond 上仍有差距。

## 三、使用了哪些模型和计算资源？

| 项目 | 信息 |
|------|------|
| 基座模型 | Qwen2.5-Coder-3B（阿里开源） |
| 参考/教师模型 | 用「强教师模型」做蒸馏与伪标签（论文未点名具体型号）；难题筛选用前代 VibeThinker-1.5B 当参照 |
| 推理后端 | vLLM；评测参数 temperature 1.0、top-p 0.95、top-k -1 |
| SFT 超参 | 全局 batch 128，学习率 $5\times10^{-5}$ 余弦退火到 $8\times10^{-8}$；第一阶段 5 轮 + 第二阶段 2 轮 |
| RL 上下文窗口 | 单一 64K 长上下文，全程 on-policy |
| GPU 型号 / 训练时长 / 成本 | **论文正文未披露** |

⚠️ 关于「成本」需特别澄清：网络上广泛流传的「约 7,800 美元、约 3900 张 H800 GPU 小时」其实是**前代 VibeThinker-1.5B** 的训练成本（对比 DeepSeek R1 的 29.4 万美元、MiniMax-M1 的 53.5 万美元），并非本文 3B 模型的官方数字。本篇 3B 技术报告中并没有给出具体的 GPU 型号、卡时或美元成本，请勿张冠李戴。

## 四、实验结果

模型经过严格的「数据去污染」处理，在多个独立竞赛体系上评测。VibeThinker-3B 自身成绩（数学类取 64 次生成均值的 Pass@1）：

| 基准 | 类型 | VibeThinker-3B | + CLR | 参照（大模型） |
|------|------|----------------|-------|----------------|
| AIME25 | 数学竞赛 | 91.4 | **96.7** | DeepSeek V3.2: 93.1 |
| AIME26 | 数学竞赛 | 94.3 | **97.1** | DeepSeek V3.2: 94.2 / Kimi K2.5: 93.3 / Gemini 3 Pro: 91.7 |
| HMMT25 | 数学竞赛 | 89.3 | **95.4** | DeepSeek V3.2: 90.2 |
| BruMO25 | 数学竞赛 | 93.8 | **99.2** | DeepSeek R1: 92.5 |
| IMO-AnswerBench | IMO 级 | 76.4 | **80.6** | DeepSeek V3.2: 78.3 (671B) / GLM-5: 82.5 (744B) |
| LiveCodeBench v6 | 编程 | 80.2 | — | DeepSeek V3.2: 80.8 |
| OJBench | 编程 | 38.6 | — | 超过表中所有 <14B 小模型 |
| GPQA-Diamond | 研究生级知识 | 70.2 | 72.9 | DeepSeek V3.2: 78.3 等（**明显落后**） |
| IFEval | 指令遵循 | 93.4 | — | 说明强化推理没损害听话能力 |

**几个看点：**

- 在数学竞赛上，3B 的它直接和 671B、1T 的开源/闭源旗舰打成平手，加 CLR 后甚至冲进第一梯队。
- 在**没见过的新题**上泛化极强：2026 年 4–5 月的 LeetCode 周赛/双周赛，通过率 **96.1%（123/128）**，与 GPT-5.2（95.3%）、Gemini 3 Flash（96.9%）同档。这条最有说服力，因为题目是训练后才出现的，背不了答案。
- **诚实的短板**：在知识密集的 GPQA-Diamond 上只有 70.2，明显落后大模型。作者并不回避，反而用它佐证自己的假说——小模型能压缩「推理」，但压不下「海量知识」。

## 五、潜在应用与已落地应用

- **开源与许可**：模型以**宽松的 MIT 许可**发布在 [Hugging Face](https://huggingface.co/WeiboAI/VibeThinker-3B) 和 ModelScope，权重可自由商用。代码与说明见 [GitHub 仓库](https://github.com/WeiboAI/VibeThinker)。
- **端侧 / 低成本部署**：BF16 权重约 6 GB 出头，社区已放出多种 GGUF 量化版（如 Q4_K_M 约 2 GB），可用 llama.cpp 在 CPU 笔记本、甚至高端手机上慢跑；官方推荐用 vLLM 或 SGLang 推理，支持最高约 100K 输出 token。
- **适用场景**：数学解题、竞赛级编程、STEM 推理、带明确约束的指令执行等「答案可验证」的任务，特别适合对成本/延迟/隐私敏感、想本地跑推理引擎的场景。
- **不适用**：开放域百科问答、需要广博世界知识的通用任务——这是它的能力边界，作者自己也讲清楚了。

## 六、网络上的讨论与评价

这款模型在社区引发了不小的热度，也伴随着争议：

- **正面**：[VentureBeat](https://venturebeat.com/technology/why-weibos-tiny-vibethinker-3b-has-the-ai-world-arguing-over-benchmarks-again) 等媒体惊叹一个 3B 模型竟能在 AIME26 上追平 671B 的 DeepSeek V3.2、超过 Gemini 3 Pro（91.7）。[36氪](https://eu.36kr.com/en/p/3857190890378249) 以「3B 小模型在编程上媲美 Opus 4.5」为题报道。知名 AI 教育者 [Sebastian Raschka](https://sebastianraschka.com/blog/2026/vibethinker-3b-post-training.html) 评价说，这「清楚展示了优秀的数据筛选与后训练能带来多大收益」。
- **质疑（benchmark 之争）**：核心争论在于**分数是作者自测、未经独立第三方复现**。同一个模型在同一测试上，分数会因 prompt 格式、采样次数、token 预算、是否开启测试时扩展而浮动好几分——CLR 把 AIME26 从 94.3 抬到 97.1，被指代表两种完全不同的部署成本场景，不能混为一谈。
- **泛化性争议**：GPQA-Diamond 仅 70.2，说明它是「**可验证推理专才**」而非通用模型。批评者提醒：竞赛数学/编程上的高分，未必能推广到更广的通用能力。
- 综合来看，社区共识大致是：方法（数据 + 后训练 + 测试时扩展）确实漂亮且高效，但「3B 媲美万亿模型」的标题应理解为「在特定可验证任务上」，需保持谨慎。

> 注：本文 HuggingFace 论文页的具体投票数暂未查到准确数字（frontmatter 中暂记为 0）。

## 七、思维导图

```mermaid
mindmap
  root((VibeThinker-3B 小模型逼近顶级推理))
    研究背景与问题
      主流依赖参数Scaling Laws堆大模型
        顶级推理集中在数十亿至万亿参数
        普遍认为3B以下小模型搞不定复杂推理
      延续前作VibeThinker-1.5B
        1.5B证明小模型能完整推理
        3B追问进入顶级梯队的参数阈值
    核心方法 Spectrum-to-Signal SSP
      SFT阶段 铺光谱
        Diversity-Exploring Distillation 多样性探索蒸馏
        Multi-path Reasoning Distillation 保留多条推理轨迹
        两阶段课程学习 5epochs+2epochs
          难题筛选 推理<5K丢弃 错误率<0.75剔除
        多级质控 N-gram+LLM+沙箱执行
        领域专家模型参数合并
      RL阶段 放信号
        MGPO MaxEnt-Guided Policy Optimization
          按最大熵加权 聚焦p(q)≈0.5能力边界题
          GRPO-style clipped objective
          全程on-policy 缓解train-inference mismatch
        多领域可验证RL Math到Code到STEM
        Single 64K长上下文 不做渐进式截断
        Long2Short Math RL 零和奖励重分配 缩短冗余
      Offline Self-Distillation 离线自蒸馏
        Learning-potential Filtering 学习潜力分数
        domain length buckets内部排序
      Instruct RL 指令对齐
        rule-based validators + rubric reward models
      CLR 测试时扩展
        K=32候选 M=5关键论断 模型自验证
        非线性可靠性分数 r_k=(均值)^M 重罚逻辑漏洞
        可靠性加权聚合选答案
    实验设计与结果
      基座Qwen2.5-Coder-3B vLLM推理
      数学竞赛 追平671B/1T旗舰
        AIME26 94.3 加CLR到97.1
        BruMO25 93.8 加CLR到99.2
        HMMT25 89.3 加CLR到95.4
      编程 LiveCodeBench v6 80.2
        LeetCode新赛OOD泛化 96.1%
      短板 GPQA-Diamond 仅70.2 知识落后
      IFEval 93.4 推理增强未损听话
    理论与影响
      Parametric Compression-Coverage Hypothesis
        推理可压缩成紧凑reasoning core
        知识需广覆盖参数
      Reasoning-Knowledge Decoupling Paradigm
      小模型是与Scaling互补的研究路线
      MIT开源 端侧低成本部署 GGUF量化
```
