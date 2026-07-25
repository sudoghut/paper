---
title: "论文综述：DeepSearch-World：可验证环境中深度搜索智能体的自蒸馏方法"
originalTitle: "DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment"
originalUrl: "https://arxiv.org/abs/2607.07820"
authors: "Xinyu Geng, Xuanhua He, Sixiang Chen, Yanjing Xiao, Fan Zhang, Shijue Huang, Haitao Mi, Zhenwen Liang, Tianqing Fang, Yi R. Fung"
institution: "HKUST, Tencent, HKUST(GZ)"
hfVotes: 89
publishDate: "2026-07-13"
reviewDate: "2026-07-25"
tags: ["deep search agent", "self-distillation", "verifiable environment", "tool-use LLM", "agentic training"]
description: '提出42万题可验证维基百科环境与自蒸馏训练框架，让9B模型无需依赖更强模型蒸馏即可自我进化为深度搜索智能体'
---

## 一、论文是干什么的？

想象你要培养一名侦探，让他学会在图书馆里查资料、连线索、破案子。传统的两种培养方式都有问题：第一种是「抄作业式」教学（监督微调 SFT）——找一个更厉害的老侦探，把他破案的完整笔记抄给新人看，新人的能力上限被老侦探的水平和笔记的多样性死死限制住；第二种是「放养式」教学（强化学习 RL）——让新人自己随便查案，只在案子破没破的那一刻给一个「对」或「错」的分数，中间几十步的搜索、翻资料、纠错到底哪一步做得好完全不知道，学习信号非常稀疏。

这篇论文提出了第三条路：先造一个「可验证的案发现场」，再让新人一边办案一边被这个现场客观地打分，然后拿自己办得好的案子当教材来教下一个版本的自己——这就是自蒸馏（self-distillation）。具体来说，作者做了两件事：第一，构建了 **DeepSearch-World**，一个基于维基百科、包含42万道多跳问答题的确定性（deterministic）、可复现的离线搜索环境，每道题的正确线索（实体）都能被环境客观核对，而不需要靠另一个AI裁判去主观打分；第二，提出了 **DeepSearch-Evolve** 自蒸馏训练框架，让一个9B参数的模型完全在这个环境里自我练习、自我出教材、自我进化，最终不依赖任何比自己更强的闭源模型蒸馏，就在多个深度搜索基准上取得了和主流开源智能体相当甚至更好的成绩。

## 二、核心方法与创新

### DeepSearch-World：一座「可查分的案发现场」

把维基百科当成一张巨大的关系网（知识图谱），论文从种子词条出发做广度优先探索，让一个大模型（Gemini-3-Pro）沿着超链接不断走「实体随机游走」（entity-level random walk），每走一步就形成一条新的推理链条。基于这些随机游走路径，作者生成了42万道多跳问答题：题目会故意把明确的实体名字换成模糊的描述（比如把"2014年"换成"2010年代"，把"北京"换成"中国某大城市"），这个操作叫**特征模糊化**，目的是防止模型靠死记硬背蒙对答案，逼它必须真的去搜索验证。

为了让这个环境可验证、可复现，论文没有直接接入真实互联网，而是把约1000万条维基百科词条整理成一个离线语料库，配了两个「假装是真实网页」的工具：`web_search_wiki`（用 BM25 检索引擎模拟搜索引擎，返回标题、摘要、链接）和 `visit_wiki`（用 SQLite 偏移索引模拟点开一个网页，返回全文）。因为是离线、确定性的，同一个问题、同一个操作，得到的结果永远一样——这就好比给侦探一个「有标准答案的案发现场沙盘」，而不是让他去乱糟糟、随时会变的真实城市里找线索。

更关键的是**过程级验证**：环境预先知道每道题需要找到哪些「目标实体」，侦探每查一次资料，环境都会核对这次返回的信息里有没有命中还没找到的目标实体，命中了就记一笔（entity-level verification）。查错了、没查到，环境会给出「环境接地的反思」（environment-grounded reflection）——第一次失败给个模糊提示，反复失败会给出更明确的线索，引导侦探换个搜索词重新试。这样一来，每一步搜索是好是坏都有客观依据，而不需要额外找一个AI来主观评判。

### DeepSearch-Evolve：生成轨迹 → 筛选成功案例 → 混合数据 → 微调 的自我进化循环

有了这个可打分的沙盘，DeepSearch-Evolve 用一个「脚手架教师」（scaffold teacher）先跑一遍：这个教师按照 **计划（Plan）→行动（Act）→结束（End）** 三阶段工作，随身带着一本「工作备忘录」，里面有四栏可写内容：已完成的线索（completed_list）、待办清单（todo_list）、经验教训（experience）、已确认的信息（information）。每一步的思考过程由三部分拼接而成：当前进度状态改写、环境反思改写、以及本次动作的理由（对应论文公式 (2)：$\langle\text{think}\rangle_t = P_t \oplus R_t \oplus A_t$）。

整个自我进化过程是这样一个循环，每一轮（round）大约包含：

1. **生成轨迹（Teacher Generation）**：当前这一版模型 $\pi_{\theta_R}$ 扮演教师，对一批新问题（每批1万道）跑上面的 Plan-Act-End 流程，最多走30步，产生带搜索、阅读、反思、纠错的完整轨迹。
2. **拒绝采样与质量过滤（Rejection Sampling & Quality Filtering）**：先用另一个模型（Qwen3.5-9B）判断最终答案对不对，只留下答案正确的轨迹（拒绝采样，RS）；再用一个「工具调用合理性」质量过滤器（QF），剔除掉证据重复、目标不对齐、推理不连贯的轨迹，即使答案凑巧对了也会被删掉。凑够4000条通过筛选的轨迹就触发一次训练。
3. **脚手架转标准格式（Scaffold-to-ReAct Distillation）**：教师专属的计划/备忘录提示词只是用来"教学"的辅助工具，学生模型不应该依赖这些外部提示。所以要把保留下来的轨迹，去掉阶段专用提示词和反思标记，改写成标准的 ReAct 格式（思考-工具调用-观察-思考-回答），把"计划、记忆追踪、纠错"这些能力内化进模型自己的思考过程里，而不是靠外部脚手架撑着。
4. **学生自蒸馏微调（Supervised Self-Distillation Update）**：用这些转换好的 ReAct 轨迹对模型做监督微调（SFT），微调后的模型 $\pi_{\theta_{R+1}}$ 变成下一轮的教师，如此往复。

打个比方，这就像一个学生自己刷题库：先自己做题、自己对答案，把做对且思路清晰的解法整理成规范笔记，用这些笔记复习提升，然后用提升后的自己再去刷下一批题——完全不需要班上有个学霸（更强的闭源模型）来抄他的解题过程。由于轨迹生成和模型训练是异步进行的（生成永远不会被训练阻塞），当训练进度落后于新一批轨迹时，论文用"重要性采样"（importance sampling），并加入衰减因子 $\gamma=0.5$，让最新一轮的数据权重更高，同时保留部分早期数据防止模型"遗忘"之前学到的行为。

论文正文中报告主实验使用了 **11轮**自演化训练，而附录中给出的默认训练超参数配置写的是 $R=15$ 轮——这是原论文正文与附录的一处细节差异，如实注明供读者留意。此外，为缩小"离线沙盘"和"真实互联网"之间的执行差异，作者还在1600条真实工具调用实例上，用 GRPO 做了一轮轻量强化学习微调（调用 Google SerpAPI 搜索和 Jina 抓取网页）。

消融实验证明每个环节都有用：如果去掉拒绝采样和质量过滤，SearchQA 分数从58.2掉到46.4；如果把"脚手架轨迹"直接拿来训练而不做反思改写（即保留原始的 `[REFLECTION]` 标记等噪声），DeepSearch-Val 分数从31.9骤降到15.7左右，说明"去噪转格式"这一步至关重要。

## 三、使用了哪些模型和计算资源？

- **底座模型**：论文明确使用开源模型 **Qwen3.5-9B-Instruct** 作为骨干模型，DeepSearch-World-9B 就是在这个底座上通过上述自蒸馏循环训练得到的，训练全程只用了模型自己生成并经环境验证的轨迹，**没有**从任何更强的闭源模型（如 GPT、Gemini、Claude 等）蒸馏数据。（用于质量筛选阶段判断答案对错的 Qwen3.5-9B 也是同款开源模型，而非借用了更强模型的判断力。）
- **GPU 型号与数量**：论文附录 B.4「异步训练效率」部分明确写道，吞吐量测试是在**单个8卡 Nvidia H20 节点**上完成的（用 vLLM 做轨迹生成，张量并行 TP=4 或 TP=8）；而完整的端到端自演化训练（论文举例是5轮）是在**单个2节点的 H20 集群**上完成的。论文没有给出精确的 H20 总卡数（如是否为2×8=16卡），只写了"single 2-node H20 cluster"这一描述，如实按原文转述。
- **训练耗时**：附录原文写道，"a complete 5-round self-evolving loop completes in approximately 2–3 days on a single 2-node H20 cluster"，即5轮自演化闭环大约耗时**2-3天**。论文主实验用了11轮（一说15轮，见上节说明），完整11/15轮所需的总时长论文未直接给出，需按比例推算，原文未明确说明。
- **训练超参数**（附录 B.3）：每批生成 $B=10{,}000$ 道问题；凑够 $C_{\min}=4{,}000$ 条正确轨迹触发一次训练；重要性采样混合后目标轨迹数 $N_{\mathrm{target}}=4{,}000$；衰减因子 $\gamma=0.5$；用 LlamaFactory 训练，每轮1个 epoch，学习率 $5\times10^{-6}$，余弦学习率调度，10% warmup，DeepSpeed ZeRO-2，BFloat16 精度，最大序列长度32,768；智能体最多走30步（$T_{\max}=30$），滑动窗口保留最近3步细节（$w=3$），单条观测截断至8,192字符。
- **推理时每题的平均耗时/成本**：论文**没有**给出每道题的平均推理秒数或美元成本这类逐题指标，只报告了聚合吞吐量数据：在单个8卡 H20 节点上，用 vLLM 每个生成批次可处理约1万道问题；训练（对4000条轨迹做1个 epoch 的 SFT）能在下一批轨迹生成的同时完成，实现生成与训练的流水线重叠。因此这部分严格按论文原文，如实标注为"论文未明确说明"。

## 四、实验结果

论文在7个搜索密集型基准上评测：BrowseComp（英文高难度网页检索）、BrowseComp-ZH（中文版）、HLE（专家级学术推理）、GAIA-Text（真实世界多步任务）、xbench-DeepSearch（中文深度搜索）、HotpotQA（多跳问答）、SearchQA（综合检索问答，用于消融）。

主结果对比（部分数据，单位：%）：

| 模型 | BrowseComp | BrowseComp-ZH | HLE | GAIA | xbench | HotpotQA |
|---|---|---|---|---|---|---|
| OpenAI Deep Research（闭源） | 51.5 | 42.9 | 26.6 | 67.4 | - | - |
| OpenAI o3（闭源） | 49.7 | 58.1 | 20.2 | 70.5 | 65.0 | - |
| WebSailor（开源） | 6.7 | 14.2 | 12.8 | 37.9 | 34.3 | 92.8 |
| Marco-DR（开源） | 31.4 | 47.1 | - | 69.9 | 42.0 | - |
| MiroThinker-v1.0（开源） | 31.1 | 40.2 | 21.5 | 66.4 | 34.0 | - |
| Qwen3.5-9B-Instruct（底座，未训练） | 7.4 | 13.5 | 16.7 | 23.9 | 20.0 | 45.3 |
| **DeepSearch-World-9B（本文）** | **31.2** | 36.4 | 25.7 | 61.5 | 49.0 | **93.4** |

可以看到，DeepSearch-World-9B 相比自己的底座 Qwen3.5-9B-Instruct 在各项指标上都大幅提升：BrowseComp +23.8、BrowseComp-ZH +22.9、HLE +9.0、GAIA +37.6、xbench +29.0、HotpotQA +48.1；同时它的表现已经追平甚至超过了多个专门用强模型蒸馏数据训练的开源智能体（如 Marco-DR、MiroThinker-v1.0），而它完全没有用到任何强模型的蒸馏信号。论文也坦承 BrowseComp-ZH 分数偏低是预期之中的，因为训练数据只用了英文轨迹，中文表现的提升更多来自跨语言的部分迁移。

行为分析也很直观：底座模型 Qwen3.5-9B-Instruct 平均只走4.7轮就草草给出答案（经常过早下结论），只调用0.9次网页阅读；而 DeepSearch-World-9B 平均能坚持18.0轮，调用5.4次网页阅读，展现出更强的"锲而不舍查资料"的习惯。用 LLM 裁判从计划、记忆、推理、纠错、证据收集五个维度打的"进阶能力分"（Advanced Capability Score）上，底座模型只有19%，DeepSearch-World-9B 达到70%。

消融实验证明各模块都有贡献：拒绝采样（RS）是最关键的一环，去掉后 SearchQA 从58.2掉到46.4；质量过滤单独作用较小，但和拒绝采样叠加效果最好。轨迹转格式环节里，"反思改写"最重要——去掉它会让 DeepSearch-Val 分数从31.9骤降到16.7甚至15.7，因为未经改写的原始反思标记（如 `[REFLECTION]`）会干扰模型自己的思考过程分布（这个"w/o both"数值论文正文写15.7，但其自身 Table 3 写的是14.8，属于原文正文与表格间的细微差异，非本综述编造）。此外，论文比较了不同数据规模下的自演化效果：420K 全量题库比只用100K 题库能达到更高的验证集分数平台期，说明训练数据的多样性比重复刷同一批题更重要。

## 五、潜在应用与已落地应用

**潜在应用方向：**
- **AI 搜索助手 / Deep Research 产品**：这类"能自己判断信息是否足够、会反复搜索验证、能从失败中恢复"的智能体，正是当下 OpenAI Deep Research、Google Deep Research、各类"深度研究"产品背后所需要的核心能力，本文方法可以用更低成本（不依赖强模型蒸馏）训练出同类能力的开源版本。
- **企业内部知识库多跳问答**：把离线维基百科语料替换成企业自己的文档库，同样的"可验证环境+自蒸馏"思路理论上可以迁移过去，训练一个能在企业内部资料里做多跳检索的助手。
- **可验证训练环境本身作为基础设施**：DeepSearch-World 提出的"确定性、可复现、过程级可验证"环境设计思路，也可能被其他团队复用去构建别的智能体训练环境（不局限于搜索场景）。

**已落地情况：** 论文明确表示计划开源环境、420K训练题库、验证集、模型和代码。经核实，代码仓库已经**真实公开**在 [GitHub - ornamentt/DeepSearch-World](https://github.com/ornamentt/DeepSearch-World)，且不是空壳占位仓库，而是包含了完整的 DeepSearch-World 环境代码（离线维基百科检索/阅读工具）、DeepSearch-Evolve 自蒸馏训练流水线（脚手架教师、拒绝采样、质量过滤、Scaffold-to-ReAct 转换、LlamaFactory SFT 集成）以及数据构建脚本；作者还在 Hugging Face 上发布了420K训练数据和验证集。整体来看，论文承诺的开源工作在本次综述撰写时已经基本兑现，处于可复现研究的状态，但尚未看到围绕它构建的下游商业产品或应用案例。

## 六、网络上的讨论与评价

在 Hugging Face 论文页面上，这篇论文获得了 **89 个点赞**。页面下方能看到的实质性讨论不多，其中一条来自用户 "O96a" 的评论质疑了该方法的实际落地效果，大意是：**确定性环境**（deterministic environment）**对训练来说是优点，但在实际生产环境部署时，恰恰是这类"环境"最先出问题的地方**（原话："the deterministic environment is a strength for training but it's also the thing that usually breaks first in prod"）——即离线、确定性的维基百科沙盘和嘈杂多变的真实互联网之间存在明显的 sim-to-real 差距，这也是论文自己在"局限性"部分承认的问题之一（当前环境只覆盖维基百科，领域和真实网络的多样性有限）。

除此之外，经过检索 Twitter/X、Reddit、Hacker News、知乎、公众号等平台，**暂未搜索到围绕这篇论文的进一步公开讨论**（论文于2026年7月发布，属于较新的工作，社区讨论可能仍在积累中）。如后续搜索到更多讨论，可再补充。

## 七、思维导图

```mermaid
mindmap
  root((DeepSearch-World 自蒸馏深度搜索智能体))
    研究背景与问题
      现有自演化范式的局限
        SFT依赖固定教师轨迹-受限于backbone能力
        RL式GRPO仅有稀疏轨迹级奖励-信用分配困难
        OPSD在噪声教师分布下不可靠-长程工具决策缺乏过程监督
      本文核心主张
        长程agent需要可验证环境暴露过程级监督
    DeepSearch-World可验证环境
      数据构造
        Wikipedia超链接图G=(V,E)
        entity-level random walk H-hop采样
        Gemini-3-Pro做BFS知识树构建
        feature fuzzification防止记忆-强制多跳检索
        420K QA任务-377条DeepSearch-Val验证集
      离线工具
        web_search_wiki-Lucene BM25索引-Pyserini
        visit_wiki-SQLite offset索引-字节偏移随机访问
        keyword/caption/contents三字段索引
        约1000万条Wikipedia语料
      过程级验证机制
        目标实体集合Ti-completed set St
        entity命中即验证进度-order-free verification
        环境接地反思机制 environment-grounded reflection rt
        搜索失败触发分级线索提示
    DeepSearch-Evolve自蒸馏框架
      Scaffold Teacher三阶段
        Plan-初始化progress state
        Act-最多Tmax=30步交互循环
        End-证据充分或预算耗尽生成答案
        working memory四字段-completed_list/todo_list/experience/information
      think块构造公式2
        Pt进度状态改写
        Rt反思改写为自我纠正
        At动作理由
      自演化训练循环
        Teacher Generation-当前策略piθR生成轨迹
        Rejection Sampling-Qwen3.5-9B判定答案正误
        Quality Filtering-工具调用合理性LLM评估
        Scaffold-to-ReAct转换-剔除脚手架专属提示
        Supervised Self-Distillation Update-SFT训练piθR+1
      优化目标
        LSFT公式3-KL(δyt‖πθ)硬标签蒸馏
        LOPSD公式4-KL(q^t‖π^t_θ)软标签token级蒸馏
        evolving SFT取代全on-policy分布匹配保证稳定性
      异步训练与数据混合
        重要性采样处理生成滞后于训练
        衰减因子γ=0.5-优先近期数据兼顾防遗忘
        11轮-附录超参数表写15轮-正文附录存在差异
        真实工具GRPO微调-1600实例-Google SerpAPI+Jina缩小sim-to-real差距
    实验设计与结果
      基准与Baseline
        BrowseComp/BrowseComp-ZH/HLE/GAIA/xbench/HotpotQA/SearchQA
        闭源对比OpenAI Deep Research-o3
        开源对比WebSailor/Marco-DR/MiroThinker-v1.0/PokeeResearch等
      主要指标结果
        DeepSearch-World-9B对比backbone提升
        BrowseComp31.2-GAIA61.5-HotpotQA93.4
        跨6项基准平均+28左右提升
      工具行为分析
        平均交互轮数18.0对比backbone4.7
        visit调用5.4对比0.9
        Advanced Capability Score五维度70%对比19%
      消融实验结论
        RS+QF筛选-SearchQA58.2对比w/o both46.4
        反思改写关键-DeepSearch-Val31.9对比w/o15.7
        420K题库优于100K-验证集平台期更高更稳定
    理论分析与洞察
      确定性可验证环境替代部分教师监督
      order-free entity验证降低LLM裁判成本
      scaffold-to-ReAct内化planning/memory/reflection能力
      数据多样性优于数据重复暴露-防止过拟合窄分布
    影响与展望
      潜在应用-AI搜索助手/Deep Research产品/企业知识库
      已开源GitHub仓库ornamentt/DeepSearch-World
      局限性-仅覆盖Wikipedia领域-evolving SFT非RL/OPSD
      未来方向-扩展至更广知识源-融合RL式更新提升泛化
```
