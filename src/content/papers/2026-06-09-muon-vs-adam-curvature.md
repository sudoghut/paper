---
title: "论文综述：为什么Muon优于Adam——来自曲率的几何解释"
originalTitle: "Why Muon Outperforms Adam: A Curvature Perspective"
originalUrl: "https://arxiv.org/abs/2606.04662"
authors: "Shuce Wang, Fengzhuo Zhang, Jiaxiang Li, Dirk Bergemann, Zhuoran Yang"
institution: "新加坡国立大学、耶鲁大学、明尼苏达大学"
hfVotes: 51
publishDate: "2026-06-09"
reviewDate: "2026-06-09"
tags: ["optimizer", "Muon", "Adam", "curvature", "LLM-training", "optimization-theory"]
description: "首次用二阶泰勒展开和归一化方向锐度（NDS）指标，从曲率角度严格证明Muon优于Adam的根本原因：Adam每步遭遇的归一化方向锐度是Muon的1.76倍，导致更大的曲率惩罚，而数据不均衡程度越高这一优势越显著，已在Kimi K2等万亿参数模型上落地验证"
---

## 一、论文是干什么的？

**Adam**（2014年）是深度学习最广泛使用的优化器，对每个参数独立自适应步长——梯度大的方向步长小，梯度小的方向步长大。**Muon**（2024年 Keller Jordan 在 NanoGPT 速度竞赛中发明）通过"牛顿-舒尔茨正交化"将更新矩阵的奇异值拉平，把权重矩阵当成整体几何对象处理。

Muon 已在 Kimi K2（万亿参数MoE）、GLM-4.5、INTELLECT-3 等模型中落地，月之暗面报告称 Muon 只需 AdamW 约52%计算量即可达到同等效果。但"**为什么 Muon 更好**"此前没有令人信服的理论解释——这正是本文的核心贡献。

## 二、核心方法与创新

**核心工具：归一化方向锐度（Normalized Directional Sharpness，NDS）**

论文用二阶泰勒展开把每步损失下降量拆分为两部分：

$$\Delta L \approx \underbrace{-\eta \langle g, d \rangle}_{\text{一阶项（梯度对齐）}} + \underbrace{\frac{\eta^2}{2} d^\top H d}_{\text{二阶项（曲率惩罚）}}$$

其中 $g$ 是梯度，$d$ 是更新方向，$H$ 是 Hessian 矩阵。NDS 定义为：

$$\text{NDS}(d) = \frac{d^\top H d}{\|d\|^2}$$

**四大核心发现：**

1. **Adam 的 NDS 平均是 Muon 的 1.76 倍**：两者步长（更新幅度）接近，但 Adam 每步都踩在曲率更高的地形上，被"曲率折扣"吸走更多能量
2. **Muon 的优势来自更小的曲率惩罚，而非更大的梯度对齐**：Adam 倾向于沿梯度最大方向更新，这些方向往往也是曲率最高的方向；Muon 通过正交化均匀分摊能量，避开高曲率区域
3. **数据不均衡放大 Muon 的优势**：Zipf 分布不均衡程度从 $s=0$ 提高到 $s=1$ 时，Muon 的 NDS 优势扩大约1.8倍——而真实语料库正是 Zipf 分布
4. **训练中后期优势集中于层内曲率**：Muon 的曲率优势越来越集中在 within-layer 项，而非层间交叉项

**直觉比喻**：Adam = 跟着感觉冲最陡的下山路，但容易踩进深沟；Muon = 用平均分配的步伐走，绕远路走缓坡，综合效率反而更高。

## 三、使用了哪些模型和计算资源？

- **主实验模型**：124M 参数 NanoGPT（12层，12头，隐藏维度768）
- **合成数据实验**：9M 参数 NanoGPT
- **主实验数据集**：FineWeb（英文网络文本）
- **合成数据集**：Zipf-PCFG（可控不均衡概率上下文无关文法）
- **Hessian-vector products**：每500步计算一次（代价是参数量平方级，限制了实验规模到124M）
- **GPU 型号/数量/训练时长**：论文未提及（NanoGPT规模通常用A100/H100单机）

## 四、实验结果

论文主要通过测量 NDS 比值和训练过程中各指标变化来验证理论：

| 指标 | Adam | Muon | 比值 |
|------|------|------|------|
| 平均 NDS | 较高 | 较低 | Adam/Muon ≈ **1.76** |
| 梯度对齐（一阶项） | 相近 | 相近 | ≈1.0 |
| 每步净损失下降 | 较低 | 较高 | Muon优势显著 |

数据不均衡敏感性：Zipf $s=1$ 时 Muon 的 NDS 优势比 $s=0$ 时扩大约1.8倍。

**已落地模型验证（外部独立报告）：**
- Kimi K2（月之暗面，万亿参数MoE）：使用 MuonClip，15.5T Token训练，全程零 loss spike
- GLM-4.5（智谱AI）：Muon 预训练
- NVIDIA NeMo：官方提供 Muon 工业级实现（emerging_optimizers 包）

## 五、潜在应用与已落地应用

1. **通用 LLM 预训练**：替代 AdamW 成为标准，可降低约50%训练成本（已在 Kimi K2 等多个产业级模型验证）
2. **数据不均衡场景**：Muon 在真实网络文本（Zipf 分布）上优势更显著
3. **Transformer 全连接层/注意力投影层**：Muon 主要针对2D权重矩阵，嵌入层等1D参数仍用Adam
4. **分布式训练**：NVIDIA 已提供工业级实现，通信开销约为 AllReduce 的2倍但整体效率更优

## 六、网络上的讨论与评价

Muon 是 ML 社区近两年讨论最热的优化器话题。Keller Jordan 2024年底用 Muon 打破 NanoGPT 训练速度记录后，相关讨论大量涌现；月之暗面"Muon is Scalable for LLM Training"（arXiv:2502.16982）报告在 X（Twitter）上引发大量转发，被视为"优化器领域十年来第一次真正挑战 Adam 地位的里程碑"。本论文是**首篇提供曲率机制解释**的工作，HuggingFace Papers 当日51个赞。同期相关理论工作活跃（Spectral Flattening、Newton-Muon、Mousse、AdaMuon、MuonEq等），证明研究社区对其机制的探究热情极高。主要质疑：Muon 仅适用于2D权重矩阵；实验规模限于124M NanoGPT；归因问题（多少改进来自 Muon 本身 vs 其他技巧）仍有争议。

## 七、思维导图

```mermaid
mindmap
  root((Muon vs Adam 曲率视角))
    背景
      Adam 2014年至今主流优化器
        每个参数独立自适应步长
        倾向于高梯度方向
      Muon 2024年 Keller Jordan
        牛顿-舒尔茨正交化
        奇异值拉平 均匀分摊能量
      已落地模型
        Kimi K2 万亿参数MoE MuonClip
        GLM-4.5 INTELLECT-3
        NVIDIA NeMo工业级实现
    核心理论工具
      二阶泰勒展开
        一阶项 梯度对齐
        二阶项 曲率惩罚 0.5*d_T*H*d
      归一化方向锐度NDS
        d_T*H*d / norm(d)^2
        仅衡量方向不受步长影响
    四大发现
      Adam NDS是Muon的1.76倍
        两者步长相近
        曲率惩罚是主要差异来源
      梯度对齐不是差异来源
        Adam梯度对齐未显著优于Muon
      Zipf不均衡放大优势
        s从0到1优势扩大1.8倍
        真实语料库正是Zipf分布
      中后期优势集中层内曲率
    实验设置
      124M NanoGPT
      FineWeb数据集
      Hessian-vector products 每500步
      合成数据集Zipf-PCFG
    同期理论工作
      Spectral Flattening arXiv 2605.13079
      Newton-Muon arXiv 2604.01472
      Mousse曲率感知预条件
      AdaMuon自适应版本
```
