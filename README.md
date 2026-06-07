# Huggingface Paper 论文综述

面向初学者的 AI 论文中文详细综述，持续收录 HuggingFace 高票论文。

**网站地址：** https://paper.oopus.info

## 特性

- **中文深度解读** — 每篇论文都包含背景、方法、实验、意义的完整讲解，避免术语堆砌
- **思维导图** — 每篇论文附带交互式思维导图（markmap），快速把握结构；Ctrl+滚轮缩放，双指触控缩放
- **数学公式** — 使用 KaTeX 渲染，公式清晰可读
- **无构建依赖** — 静态站点，由 Astro 生成，无需数据库

## 本地运行

```bash
npm install
npm run dev
```

## 添加论文

在 `src/content/papers/` 下新建 Markdown 文件，格式参考已有文件：

```yaml
---
title: "论文综述：..."        # 必填
originalTitle: "..."          # 必填
originalUrl: "https://arxiv.org/abs/..."  # 必填
authors: "..."                # 必填
publishDate: "YYYY-MM-DD"     # 必填
reviewDate: "YYYY-MM-DD"      # 必填
institution: "..."            # 可选
hfVotes: 100                  # 可选
tags: ["tag1", "tag2"]        # 可选，默认 []
description: "一句话摘要"     # 可选
---
```

思维导图用 mermaid mindmap 语法（缩进树形结构）写在正文的代码块中，浏览器加载时由自定义脚本（`PaperLayout.astro`）解析并转换为交互式 markmap 渲染（非标准 Mermaid 渲染器）。

## 技术栈

- [Astro](https://astro.build) — 静态站点框架
- [markmap](https://markmap.js.org) — 思维导图渲染
- [KaTeX](https://katex.org) — 数学公式
