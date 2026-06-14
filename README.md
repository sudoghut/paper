# Huggingface Paper Review

In-depth, beginner-friendly AI paper summaries in Chinese, curated from [HuggingFace Papers](https://huggingface.co/papers) based on personal reading interests.

**Website:** https://paper.oopus.info

**Note:** This is a personal reading selection — papers are chosen based on the author's interests, so image and multimodal papers are collected less frequently. Updated occasionally from daily papers and at least once a week from weekly papers.

## Features

- **Deep Chinese summaries** — each paper covers background, method, experiments, and significance; no jargon dumping
- **Interactive mind maps** — each paper includes a markmap mind map; Ctrl/⌘+scroll to zoom and drag to pan on desktop; two-finger pinch/pan on mobile
- **Math formulas** — rendered with KaTeX
- **Static site** — built with Astro; no database required

## Local Development

```bash
npm install
npm run dev
```

## Adding a Paper

Create a new Markdown file under `src/content/papers/` following the existing format:

```yaml
---
title: "论文综述：..."        # required
originalTitle: "..."          # required
originalUrl: "https://arxiv.org/abs/..."  # required
authors: "..."                # required
publishDate: "YYYY-MM-DD"     # required
reviewDate: "YYYY-MM-DD"      # required
institution: "..."            # optional
hfVotes: 100                  # optional
tags: ["tag1", "tag2"]        # optional, defaults to []
description: "一句话摘要"     # optional
---
```

Mind maps are written as Mermaid `mindmap` code blocks in the body. The browser converts them to interactive markmaps at runtime via a custom script in `PaperLayout.astro` (not a standard Mermaid renderer).

## Tech Stack

- [Astro](https://astro.build) — static site framework
- [markmap](https://markmap.js.org) — interactive mind map rendering
- [KaTeX](https://katex.org) — math formula rendering

---

# Huggingface Paper 论文综述

面向初学者的 AI 论文中文详细综述，从 [HuggingFace Papers](https://huggingface.co/papers) 按个人阅读兴趣精选收录。

**网站地址：** https://paper.oopus.info

**说明：** 这是个人阅读 HuggingFace Papers 的文选，收录篇目以个人兴趣为准，因此图像、多模态相关论文收录较少。更新频率：daily papers 不定期更新，weekly papers 至少每周更新一次。

## 特性

- **中文深度解读** — 每篇论文都包含背景、方法、实验、意义的完整讲解，避免术语堆砌
- **思维导图** — 每篇论文附带交互式思维导图（markmap）；桌面端 Ctrl/⌘+滚轮缩放、拖拽平移，移动端双指缩放/平移
- **数学公式** — 使用 KaTeX 渲染，公式清晰可读
- **静态站点** — 由 Astro 生成，无需数据库

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

思维导图用 mermaid mindmap 语法（缩进树形结构）写在正文的代码块中，浏览器加载时由自定义脚本（`PaperLayout.astro`）在客户端解析并转换为交互式 markmap 渲染（非标准 Mermaid 渲染器）。

## 技术栈

- [Astro](https://astro.build) — 静态站点框架
- [markmap](https://markmap.js.org) — 思维导图渲染
- [KaTeX](https://katex.org) — 数学公式
