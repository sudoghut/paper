---
title: "论文综述：Self-Retrospection Distillation——让智能体把「事后复盘」变成「事前远见」"
originalTitle: "Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight"
originalUrl: "https://arxiv.org/abs/2610.08077"
authors: "Haoxiang Zhang, Qinglin Chen, Hiroaki Hayashi, Zhuofeng Li, Siming Zhang, Jiaxin Zhang, Jixuan Chen, Fang Wu, Pan Lu, Silvio Savarese, Julian McAuley, Chien-Sheng Wu"
institution: "Salesforce AI Research, UC San Diego, Texas A&M University, Stanford University"
hfVotes: 135
publishDate: "2026-10-06"
reviewDate: "2026-10-10"
tags: ["reinforcement learning", "RLVR", "self-distillation", "LLM agents", "GRPO", "hindsight"]
description: 'Salesforce 等团队提出 SRD：用跑完的轨迹复盘去监督模型事前的预判，作为辅助损失叠加在 GRPO、OPSD、RLSD 上，在全对或全错的奖励无差异组里也能学到东西'
---

## 一、论文是干什么的？

先回顾一个背景：现在训练大模型智能体常用一种叫 **RLVR**（带可验证奖励的强化学习）的办法。做法很像「考试打分」：让模型对同一道题做 8 次（一组 rollout，即一组试做），由程序判卷（数学答案对不对、代码能不能通过测试），再用「这一次比同组平均好多少」来决定加强还是削弱这次的行为。这类「组内相对」的方法（如 GRPO）有个硬伤：**如果一组 8 次全做对，或者全做错，大家分数一样，「比平均好多少」就全是 0，这组数据一点梯度都没有**，通常直接丢掉重采样。论文测得，在不同模型规模下，有 37% 到 98% 的组属于这种「奖励无差异」的组。

打个比方：一个学生做了一套题，结果全错。传统做法是「全错就没什么可学的，下一套」；但其实这堆错误的做题过程里，藏着「这类题到底需要什么知识」「我容易在哪里翻车」这些宝贵信息。

这篇论文提出 **前瞻学习（prospective learning）**：不只用事后经验去改「下一步怎么做」，而是用事后经验去教模型「动手之前就该预料到什么」。具体方法叫 **Self-Retrospection Distillation（SRD，自我复盘蒸馏）**：同一个模型分饰两角，学生在动手前只看题目，写下「我预计会踩哪些坑」；老师（同一模型的冻结副本）看过完整的做题过程和标准答案后，再写一遍「这题的坑是什么」；然后让学生的写法向老师靠拢。就像考完试后看着错题本，反过来训练自己「下次拿到题的第一眼就想到这个坑」。

这是 Salesforce AI Research 联合 UC San Diego、Texas A&M、Stanford 的工作，代码见 [GitHub 仓库 SalesforceAIResearch/SRD](https://github.com/SalesforceAIResearch/SRD)。

## 二、核心方法与创新

### 1. 背景：三种「利用经验」的方式

论文把智能体与环境交互的一条轨迹记为 $\tau = (a_1, o_1, \dots, a_T, o_T)$，其中 $a_t$ 是动作，$o_t$ 是观察。三种范式的区别如下：

| 范式 | 事后信息怎么用 | 监督的对象 |
|---|---|---|
| RLVR | 压缩成一个标量奖励 | 动作策略 |
| 回顾式学习（如 OPSD 自蒸馏） | 保留结构化的事后信息（正确解、技能总结等）当老师的额外提示 | 动作策略（让下一步更准） |
| 前瞻式学习（本文） | 保留结构化事后信息 | 事前的预判（foresight） |

GRPO 的组内优势为 $A(\tau_i) = (r_i - \mu_r)/\sigma_r$。一旦组内奖励全相同，所有 $A(\tau_i)=0$。这就是论文反复强调的「奖励静默」盲区。

### 2. 前瞻学习的形式化

对任务 $x$ 和环境上下文 $e$，模型动手前的预判分布是 $p_{\mathrm{fore}} = \pi_\theta(\cdot \mid x, e)$（式 4）；拿到事后信息 $z_{\mathrm{hind}}$ 后，老师给出更有信息量的分布 $p_{\mathrm{hind}} = \pi_{\bar\theta}(\cdot \mid x, e, z_{\mathrm{hind}})$（式 5），其中 $\bar\theta$ 表示停止梯度的老师。前瞻学习的目标是让预判去逼近事后的判断：

$$
\mathcal{L}_{\mathrm{pro}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, \tau \sim \pi_\theta}\left[ D(p_{\mathrm{hind}} \,\|\, p_{\mathrm{fore}}) \right]
$$

注意它不直接优化动作分布，而是训练「事前预判」这件事本身。

### 3. SRD：两种预判视角

SRD 定义了两种预判：**Knowledge（知识）**，描述交互中可能需要什么知识；**Pitfall（陷阱）**，描述可能遇到什么失败。对同一道题的一组 $G$ 条 rollout，奖励决定每条轨迹走哪个视角：成功（$r_i = 1$）用 Knowledge，失败（$r_i = 0$）用 Pitfall。所以论文说：奖励决定的是「教什么样的一课」，而不是「教不教」。

每条轨迹的特权后验信息记为 $f_i = (\tau_i, y^*, \epsilon_i)$，即轨迹、标准答案和错误标注。学生和老师的分布分别是：

$$
p^{i}_{\mathrm{fore}} = \pi_\theta(\cdot \mid x, e, c_i), \qquad p^{i}_{\mathrm{hind}} = \pi_{\bar\theta}(\cdot \mid x, e, f_i, c_i)
$$

其中 $c_i$ 是对应视角的提示指令。学生先自己采样一段预判文本 $z_{\mathrm{fore},i}$，老师在看到同样前缀的同时还能看到 $f_i$，在每个 token 位置上对齐两者的分布：

$$
\mathcal{L}_{\mathrm{SRD}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\sum_{l=1}^{L_i} D\left(\pi_{\bar\theta}(\cdot \mid x, e, f_i, z^{\mathrm{fore}, i}_{<l}) \,\|\, \pi_\theta(\cdot \mid x, e, z^{\mathrm{fore}, i}_{<l})\right)\right]
$$

散度 $D$ 实际用广义 Jensen-Shannon 散度 $\mathrm{JSD}_\beta$（$\beta = 0.5$）。老师是当前策略的停止梯度副本，实现里用指数滑动平均（EMA）自老师，更新率为每个优化步 0.05。

比喻：学生交的「预习笔记」和老师看过答案后写的「复盘笔记」逐字对比，差在哪就改哪。

### 4. 即插即用的辅助损失

SRD 不替换原有训练目标，只是加一项：$\mathcal{L}_{B+\mathrm{SRD}} = \mathcal{L}_B + \lambda \mathcal{L}_{\mathrm{SRD}}$，其中基础目标 $B$ 可以是 RLVR（GRPO）、OPSD（在线自蒸馏）、RLSD（带自蒸馏重加权的 RLVR）三者之一，主实验中 $\lambda = 0.01$。预判文本只是训练目标，推理时不必生成。

### 5. 为什么只蒸馏 Pitfall？

主表的所有 +SRD 行都只蒸馏 Pitfall 通道。论文的研究问题 RQ1 显示：在 OPSD 之上再加 Knowledge 通道并没有更好，只在个别设置中略胜（例如 AIME26 在 4B 上 67.50 对 67.08），却是唯一会低于基线的设置，同时 rollout 更慢（rollout 占一个训练步的 78% 到 87%）。训练曲线里 Pitfall 通道的散度下降了 47% 和 30%，Knowledge 通道基本不动（约 0.0045 和约 0.003）。所以作者把「只蒸馏 Pitfall」作为更便宜更稳的默认选择。

### 6. 理论视角：为什么奖励方向会「静默」

附录 A 给出推导：一组 $G$ 条 rollout 里出现混合结果的概率是 $q_G(p) = 1 - p^G - (1-p)^G \le G\,\pi(p)$，其中 $\pi(p) = \min\{p, 1-p\}$。奖励方向要拿到 $n$ 个有效组，期望至少要 $n/\pi(p)$ 次环境 rollout，也就是成功率越接近 0 或 1，代价越高；而前瞻方向没有这种「必须是混合组」的结构性门槛。作者强调这是结构上的差异，并不保证前瞻梯度一定非零。

## 三、使用了哪些模型和计算资源？

**被训练的模型：** Qwen3.5-4B 和 Qwen3.5-9B（开启 thinking 模式）。RQ3 的奖励无差异分析在 4 个 Qwen3.5 规模上做（正文图中涉及 2B、9B、35B 等规模，4 个规模的完整列表以论文图 4 为准，文本未逐一列出）。官方 GitHub 说明支持 Qwen3.5-4B、9B、35B-A3B。

**评判与辅助模型：** 评测与奖励中，LLM-as-judge 只是兜底：数学先做符号匹配，搜索先做 exact match，只有在匹配失败或参考答案是自由描述时才调用；BrowseComp-Plus 以评判模型为主要评分者。三者使用的评判模型均为 gpt-5.6-luna（OpenAI）。在构造 HotpotQA 与 2Wiki 的评测子集时，还用 Qwen3-8B（Search-R1 脚手架）做难度校准，保留 pass@8 高于 0% 且低于 30% 的题。

**计算资源：** 单节点 8 张 NVIDIA H200 GPU。GitHub 的快速上手写明推荐 1 节点 8 张 H200（141 GB），8 张 H100 或 8 张 A100-80GB 也可以，但需调低每卡最大 token 数；技术栈为 Megatron-LM、SGLang、Ray，用 enroot 启动。

**训练设置（论文表 4、表 5 与 C.2）：**

| 项目 | 取值 |
|---|---|
| 每步 prompt 数与每个 prompt 的 rollout 数 | 16 个 prompt，每个 8 条，共 128 条轨迹（动态采样重采样前） |
| 每轮最大响应长度 | 8192 token；预判（foresight）最大 2048 token |
| 训练最大工具调用轮数 | 8 |
| 采样 | temperature 1.0，top p 1.0 |
| 优化器 | Adam，学习率 1e-6 恒定，warmup 10 步，weight decay 0.1，betas 为 0.9 与 0.98 |
| 蒸馏 | top-k 100，JSD $\beta=0.5$，散度截断 2.0，重要性采样截断 2.0 |
| 评测 | 每题 avg@8，temperature 1.0；数学、代码、搜索每轮最多输出 16K，智能体任务 8K |

**耗时（原文给出的部分）：**

- 每个训练步的 rollout 耗时（论文表 2，单位秒每步）：Qwen3.5-4B 在不同配置下为 197 到 267 秒，Qwen3.5-9B 为 195 到 279 秒。例如 4B 的 OPSD 原始轨迹前缀不加 SRD 是 197 秒，加 Pitfall 的 SRD 是 217 秒，加 Pitfall 与 Knowledge 是 221 秒；9B 对应为 195、229、243 秒。
- rollout 阶段占一个训练步的 78% 到 87%。
- 代码评测每个测试用例限时 6 秒；案例中的代码执行超时设置为 10 秒。
- 总训练步数、总训练时长、总 GPU 小时：暂无相关信息（文本只说明「名义步数在各对比方法间保持一致」）。
- 评测一次完整耗时：暂无相关信息。

## 四、实验结果

### 1. 评测任务

共 10 个基准，分四类：数学（AIME 2024、AIME 2026、AMO-Bench）、代码（LiveCodeBench-v6 的 functional 部分共 63 题、OJBench 共 77 题）、搜索（HotpotQA、2WikiMultiHopQA 各 100 题子集，以及更长程的 BrowseComp-Plus）、智能体（ALFWorld 的 100 个 OOD 游戏、WebShop 的 100 个保留购物指令）。训练数据：数学用 DAPO-Math-17K；代码用 LiveCodeBench 的 stdin 部分中等和困难题共 2000 条候选；搜索用 HotpotQA 和 2Wiki 各 1500 道题，本地 BM25 索引的 Wikipedia-18；ALFWorld 用 400 个训练游戏，WebShop 用 400 条训练指令。评测与训练在格式、环境、交互长度上都有偏移，用来检验泛化。

### 2. 主表（avg@8 通过率，百分比）

下表只列分类平均值，对比 Vanilla（未训练）、基础方法与加上 SRD 之后的结果（括号内为 SRD 带来的提升，单位百分点）。

**Qwen3.5-4B-Thinking**

| 方法 | 数学 Avg | 代码 Avg | 搜索 Avg | ALFWorld | WebShop |
|---|---|---|---|---|---|
| Vanilla | 18.50 | 22.11 | 42.24 | 39.25 | 75.34 |
| OPSD | 41.81 | 31.68 | 42.62 | 52.88 | 85.90 |
| OPSD+SRD | 48.11 (+6.30) | 37.52 (+5.84) | 52.85 (+10.23) | 66.62 (+13.74) | 87.46 (+1.56) |
| GRPO | 46.22 | 36.85 | 48.29 | 67.75 | 77.21 |
| GRPO+SRD | 56.25 (+10.03) | 45.18 (+8.33) | 56.34 (+8.05) | 71.00 (+3.25) | 83.13 (+5.92) |
| RLSD | 47.20 | 37.37 | 50.89 | 62.75 | 86.89 |
| RLSD+SRD | 50.97 (+3.77) | 37.88 (+0.51) | 52.38 (+1.49) | 68.50 (+5.75) | 88.14 (+1.25) |

**Qwen3.5-9B-Thinking**

| 方法 | 数学 Avg | 代码 Avg | 搜索 Avg | ALFWorld | WebShop |
|---|---|---|---|---|---|
| Vanilla | 13.03 | 29.10 | 50.50 | 48.62 | 86.82 |
| OPSD | 39.70 | 36.85 | 47.93 | 73.88 | 91.76 |
| OPSD+SRD | 56.92 (+17.22) | 42.46 (+5.61) | 58.24 (+10.31) | 77.13 (+3.25) | 88.15 (-3.61) |
| GRPO | 55.06 | 47.43 | 54.99 | 77.13 | 78.06 |
| GRPO+SRD | 64.61 (+9.55) | 49.65 (+2.22) | 57.43 (+2.44) | 78.25 (+1.12) | 84.18 (+6.12) |
| RLSD | 52.78 | 41.48 | 54.31 | 66.25 | 90.95 |
| RLSD+SRD | 59.67 (+6.89) | 45.03 (+3.55) | 56.79 (+2.48) | 77.25 (+11.00) | 91.12 (+0.17) |

要点：

- 摘要宣称的「最多提升 24.2 个百分点」，对应表中 9B 的 OPSD+SRD 在 AIME 2026 上 +24.16。
- 分类平均值中只有一处回退：9B 的 OPSD 在 WebShop 上 -3.61 个百分点。逐基准看，个别单项也有小幅下降（例如 4B 的 RLSD 在 OJBench 上 -0.97，9B 的 GRPO 在 2Wiki 上 -0.51）。
- 纯自蒸馏（OPSD）不稳定：9B 上 OPSD 在 HotpotQA、2Wiki、LCB-v6 上分别比未训练模型低 5.50、3.76、2.18 个百分点，数学平均还出现「9B 比 4B 更差」（39.70 对 41.81）；加上 SRD 后三者都回到基线之上（分别高 10.75、11.75、8.54 个百分点），数学随规模也恢复单调。
- 代码训练用 stdin 格式、评测用 functional 格式，SRD 仍让 4B GRPO 平均提高 8.33 个百分点。ALFWorld 未见环境上，所有基础目标和规模组合都有提升，范围 1.12 到 13.74 个百分点。BrowseComp-Plus 需要 10 倍以上的工具调用，SRD 在 6 个设置中有 5 个提升，最多 9.52 个百分点。

### 3. 奖励无差异组的实验（RQ3）

在只训练代码域、关闭动态采样的设置下，4 个 Qwen3.5 规模上被丢弃的预算比例为 98.0、39.1、37.0、41.3（呈 U 形），全错组从 98.0% 单调降到 13.4%，全对组从 0.0% 升到 27.9%，可用预算占比最高只有 63%。最极端的 2B：98% 的组全错，GRPO 峰值训练成功率仅 1.6%，最终 0.0%；同预算加上 SRD 后训练成功率达到 60.6%。注意这是训练成功率，不是测试集成绩。

### 4. 消融与分析

- **Pitfall 与 Knowledge（RQ1）：** 在 OPSD 之上，Pitfall 在 AIME 2026 的原始轨迹前缀上带来 3.3（4B）和 1.7（9B）个百分点提升，在 hindsight 前缀上带来 3.3 和 6.7；加 Knowledge 后没有明显更好。
- **词汇层面（RQ2）：** 在代码验证集上，SRD 让每条 rollout 多出约 8.6 个连接性文字、3.7 个计划性词语（如 we、need、think）、1.4 个点出问题结构的词（key、insight、constraint），同时减少 6.7 个行内数学符号、4.3 个标点、4.1 个数字、3.0 个变量名；总和只变化 0.6 个 token，工具调用语法几乎不变（+0.03）。作者的解读是：把有限的预算从「行内算东西」挪到「说清问题与计划」。
- **更新方向（RQ4）：** 在留出的代码集（626992 个位置）上，GRPO+SRD 沿 GRPO 自己的方向多走了 30%（$\alpha/\|A\|=1.30$），并加上相当大小的正交分量（$\|E_\perp\|/\|E\|=0.49$）；对 OPSD 则像一个「锚」，把原本较大的位移从 1.97 缩到 0.57。
- **测试时显式给出预判（RQ5）：** 在 ALFWorld 与 WebShop 上，几乎没有额外好处（4 个检查点里有 1 个提升 5.00 个百分点，其余 3 个只变化 2、3、11 局，共 800 局），而且若预判本身错了会误导整局，所以作者认为推理时生成预判并非免费。
- **超参 $\lambda$：** 在 OPSD 与 Qwen3.5-4B 上，$\lambda$ 取 1.0、0.1、0.01、0.001 时五个基准的平均值在 63.5 到 64.2 之间，而未训练基线是 39.3；$\lambda$ 主要在数学和代码之间做取舍。此扫描用 AIME25，不是主文用的评测集。

作者明确写出的局限包括：RQ5 的比较只有两个目标、一个规模、单个种子，因此结论只是推测；位移分析只比较了最终检查点，没有看中间检查点。

## 五、潜在应用与已落地应用

**论文自己声称或暗示的方向：**

- 作为 RLVR 或自蒸馏的「即插即用」辅助损失，在奖励稀疏、模型太弱（几乎全错）或太强（几乎全对）的阶段继续榨取训练信号。
- 工具调用型推理与长程智能体（数学、代码、搜索、ALFWorld、WebShop 等），作者在 HF 评论区称，SRD 尤其有助于缓解智能体在工具脚手架上失败多于纯推理的问题（这是作者的口头概括，不是论文的单独实验结论）。
- 作者在 HF 评论区征集「除了 pitfall 与 knowledge 之外还值得蒸馏的预判」，这属于未来研究方向，不是已验证结论。

**已落地与开源情况：**

- 代码已开源：[SalesforceAIResearch/SRD](https://github.com/SalesforceAIResearch/SRD)，Apache-2.0 许可证，每个工具（代码解释器、搜索检索、WebShop、ALFWorld）可在独立容器里运行，并提供一份面向 Claude Code、Codex 等编码智能体的一键 quickstart 说明。查询时 GitHub 上约 2 个 star、0 个 fork，创建于 2026 年 9 月 17 日。
- 论文没有声称已有产品或生产环境落地，也没有发布训练好的模型权重（作者页面和 HF 论文页未列出关联模型）。

**需要谨慎的地方：** 实验只覆盖 Qwen3.5-4B 与 9B 两个规模的主表，没有在更大模型上给出主表对比；论文伦理声明也指出，蒸馏模型自己的事后判断可能强化基础模型、数据或反馈里的错误与偏见，基准提升并不等于安全、公平或可靠。

## 六、网络上的讨论与评价

- **Hugging Face 论文页：** 共 2 条评论，一条是作者（账号 IPF，即 Haoxiang Zhang）的自我介绍帖，概括了核心问题、方法和结果（包括「2B 全错组 GRPO 为 0.0%，加 SRD 达 60.6%」），并欢迎社区讨论更多类型的预判；另一条是 Librarian Bot 的自动相似论文推荐，包括 Prospective Hindsight、SIPO、PR-OPD、AHEAD 等相关工作。这两条都不是独立的第三方评价。
- **GitHub：** [官方仓库](https://github.com/SalesforceAIResearch/SRD) 的 README 写明论文预印本于 2026 年 10 月发布；暂无可见的 issue 讨论可作评价依据。
- **相关索引与镜像：** 如 [paperswithcode.co 条目](https://paperswithcode.co/paper/2610.08077)，仅是论文摘要的转载。
- **未发现公开讨论的地方：** 在 Hacker News（Algolia 搜索，分别用 arXiv 编号和标题）没有命中该论文；用网页搜索（标题加 arXiv 编号、标题加 Reddit/X/知乎/博客关键词）只返回 arXiv 与论文镜像站，没有第三方评论；GitHub 仓库搜索 arXiv 编号也没有其它项目引用；Reddit 对脚本访问（含无头浏览器重试）返回网络策略拦截，无法直接检索，所以 Reddit 上是否有讨论属于无法确认。论文发布时间很新（2026 年 10 月 6 日提交），暂无发现更多公开讨论。

**基于论文本身内容的几点观察（并非社区评价）：**

- 论文对自己的限制比较坦率，例如撤回了一个会误导的统计量（附录 E.6）、承认 RQ5 只有单种子，并强调 $\lambda=0.01$ 是为所有领域折中而选，并未针对单个领域调优。
- 实验只有两个模型规模，且主表的所有 SRD 结果都只用 Pitfall 通道；Knowledge 通道的作用在文中主要是「冗余、昂贵」，不能说明它在别的设置里一定无用。
- 训练成功率（60.6%）和评测集成绩是两回事，阅读摘要时需要区分。

## 七、思维导图

```mermaid
mindmap
  root((SRD 自我复盘蒸馏))
    研究背景与问题
      RLVR 回顾式监督 只看事后标量奖励
      GRPO 组内优势 A = r - mean 除以 std
        奖励全相同 优势为 0 整组被丢弃
        37% 到 98% 的组奖励无差异
    方法与技术贡献
      前瞻学习 Prospective Learning
        p_fore 对 p_hind 的蒸馏 式6
        监督预判而非动作分布
      两种预判视角
        Pitfall 失败组 r 等于 0
        Knowledge 成功组 r 等于 1
      L_SRD 逐 token 对齐
        停止梯度的 EMA 自老师 更新率 0.05
        广义 JSD beta 0.5 top-k 100
      辅助损失 L_B + lambda L_SRD
        lambda 0.01 可叠加 GRPO OPSD RLSD
    实验设计与结果
      Qwen3.5-4B 与 9B 8 张 H200
      10 个基准
        AIME24 AIME26 AMO-Bench
        LCB-v6 OJBench
        HotpotQA 2Wiki BrowseComp-Plus
        ALFWorld WebShop
      主要结果
        最高提升 24.2 pp 9B OPSD+SRD 的 AIME26
        4B GRPO 数学平均 46.22 到 56.25
        2B 全错组 GRPO 0.0% 对 SRD 60.6%
    理论分析与洞察
      混合组概率 q_G 小于等于 G 乘 pi(p)
      奖励方向代价随 1 除以 pi(p) 增长
      更新位移分解 对 GRPO 沿轴多走 30%
    消融与局限
      只蒸馏 Pitfall 更省更稳
      测试时显式预判几乎无增益
      仅两个规模 单种子 结论偏保守
    影响与展望
      奖励稀疏和饱和阶段的辅助监督
      探索更多类型的预判目标
```
