# 课程笔记 AI 导读与翻译器（translator）

一个面向英文课程笔记的 Python CLI 工具：
输入网页 URL 或本地 Markdown，输出结构化中文学习笔记，并尽量保持公式、代码块、链接与 Markdown 结构稳定。

## 功能概览

- 双输入模式：支持 URL 抓取（Jina Reader）与本地 Markdown 文件
- 两阶段翻译：先全局提纲/术语抽取，再按分块翻译
- 结构保护优先：保护代码块、行内代码、数学公式、链接等敏感片段
- Markdown 护栏：内置 sanitize + autofix + lint，降低渲染风险
- 学习型输出：在翻译中补充中等密度的学习批注
- 原子写入：写入失败不污染目标文件
- 可并发翻译：支持多线程分块并行
- Snapdown 处理：URL 模式下可将 `snapdown` 图块自动转换为 `mermaid`

## 快速开始

### 1. 环境要求

- Python 3.8+
- 可用的 Moonshot/Kimi API Key（必需）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量（推荐使用 `.env`）

在仓库根目录创建 `.env`：

```env
MOONSHOT_API_KEY=your_kimi_api_key
# 可选
JINA_API_KEY=your_jina_api_key
MOONSHOT_MODEL=kimi-k2-0905-preview
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
```

CLI 启动时会自动加载 `.env`。

### 4. 最短可运行示例

#### 示例 A：翻译本地 Markdown

```bash
mkdir -p output
python -m translator translate-md \
  --in documents/6.031_note1.md \
  --out output/6.031_note1.zh.md
```

#### 示例 B：从 URL 翻译

```bash
mkdir -p output
python -m translator translate-url \
  --url https://cs231n.github.io/neural-networks-1/ \
  --out output/cs231n.zh.md
```

## 命令总览

统一入口：

```bash
python -m translator --help
```

### 主要命令

| 命令 | 用途 | 关键参数 |
| --- | --- | --- |
| `translate-md` | 翻译本地 Markdown | `--in`, `--out` |
| `translate-url` | 抓取并翻译单个 URL | `--url`, `--out` |
| `translate-url-batch` | 批量翻译 URL 列表 | `--url-list`, `--out-dir` |
| `lint-md` | 检查 Markdown 结构风险 | `--in` |
| `sanitize-md` | 预清洗 Markdown 抓取噪音 | `--in`, (`--out` 或 `--in-place`) |

### 常用翻译参数（`translate-*`）

- `--max-chunk-chars`：分块上限，默认 `8000`
- `--concurrency`：并发数，CLI 默认 `1`
- `--prompt-outline-mode`：`headings`（默认，提示词更短）或 `full`
- `--prompt-glossary-mode`：`filtered`（默认，仅注入相关术语）或 `full`
- `--timeout`：URL 抓取超时秒数，默认 `30.0`

### URL 模式参数

- `--no-snapdown-mermaid`：关闭 Snapdown -> Mermaid 自动转换
- `--jina-api-key-env`：从指定环境变量读取 Jina Key，并注入为 `JINA_API_KEY`

### 批量 URL 示例

```bash
mkdir -p output/batch
python -m translator translate-url-batch \
  --url-list url.txt \
  --out-dir output/batch
```

说明：

- `url.txt` 每行一个 URL
- 空行与 `#` 开头行会被忽略
- `--out-dir` 必须已存在（命令不会自动创建）

### Markdown 质检与修复

#### 仅检查

```bash
python -m translator lint-md --in output/cs231n.zh.md
```

#### 自动修复并输出到新文件

```bash
python -m translator lint-md \
  --in output/cs231n.zh.md \
  --fix \
  --out output/cs231n.zh.fixed.md
```

#### 原地修复

```bash
python -m translator lint-md \
  --in output/cs231n.zh.md \
  --fix \
  --in-place
```

#### 先清洗再修复（推荐处理抓取原文时使用）

```bash
python -m translator sanitize-md --in raw.md --out raw.sanitized.md
python -m translator lint-md --in raw.sanitized.md --fix --out raw.cleaned.md
```

### 调试命令

- `debug-fetch`：测试 URL 抓取
- `debug-chunk`：查看分块结果（可 `--json`）
- `debug-reconstruct`：从 chunk JSON 重构文本
- `debug-protect` / `debug-restore`：测试占位保护与还原
- `debug-profile`：仅执行 Step1（全局提纲/术语）

示例：

```bash
python -m translator debug-chunk --in documents/6.031_note1.md --json
```

## 输出内容结构

生成的 Markdown 由以下部分组成（章节名为当前实现的英文标题）：

1. 一级标题：文档标题
2. `## Meta`：来源、时间戳、模型信息
3. `## Outline`：全局结构提纲
4. `## Glossary`：术语表（英中对照与备注）
5. 正文：分块翻译后的主内容

## 配置说明

### LLM 相关环境变量

| 变量名 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `MOONSHOT_API_KEY` | 是 | 无 | Kimi API Key |
| `MOONSHOT_MODEL` | 否 | `kimi-k2-0905-preview` | 模型名覆盖 |
| `MOONSHOT_BASE_URL` | 否 | `https://api.moonshot.cn/v1` | 接口地址覆盖 |
| `JINA_API_KEY` | 否 | 无 | URL 抓取鉴权（按需） |

### 运行时调优环境变量

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `TRANSLATOR_RETRY_LOG` | `1` | 重试日志开关，`0` 关闭 |
| `TRANSLATOR_STRICT_RENDERER` | `true` | 严格 markdown-it 渲染安全检查 |
| `TRANSLATOR_MAX_SAFE_LIST_DEPTH` | `1` | 列表中代码块安全深度上限 |
| `TRANSLATOR_GLOSSARY_MAX_TERMS` | `30` | 每块注入术语条目上限 |
| `TRANSLATOR_GLOSSARY_MAX_CHARS` | `2000` | 每块术语注入字符预算 |

## 架构概览（Map-Reduce 风格）

1. 读取输入：URL 抓取或本地文件加载
2. 输入清洗：修正常见抓取残留与 Markdown 异常
3. Step1 全局画像：抽取提纲与术语表
4. 分块：按标题/段落感知切块
5. Step2 分块翻译：占位保护 -> 翻译 -> 还原 -> QA
6. 拼装输出：Meta + Outline + Glossary + 正文
7. 护栏校验：autofix + lint，不安全则失败
8. 落盘：原子写入，避免部分写入损坏

## 测试

运行全部测试：

```bash
pytest -q
```

说明：

- 集成测试会在缺少 `MOONSHOT_API_KEY` 时自动跳过
- 建议在改动翻译流程、Markdown 规则或 CLI 参数后执行全量测试

## 项目结构（简版）

```text
translator/
├── src/translator/           # 主实现
│   ├── cli.py                # CLI 入口与子命令定义
│   ├── pipeline.py           # 端到端编排
│   ├── step1_profile.py      # 全局提纲/术语抽取
│   ├── step2_translate.py    # 分块翻译与恢复
│   ├── preservation.py       # 占位保护与校验
│   ├── markdown_sanitize.py  # Markdown 预清洗
│   ├── markdown_autofix.py   # Markdown 自动修复
│   ├── markdown_lint.py      # Markdown 风险检测
│   ├── chunking.py           # 分块逻辑
│   ├── jina_reader_fetcher.py# Jina 抓取与 Snapdown 提取
│   └── snapdown_converter.py # Snapdown -> Mermaid 转换
├── translator/               # 运行时包装（支持根目录 `python -m translator`）
├── tests/                    # 测试用例
├── documents/                # 示例/产物文档
├── url.txt                   # URL 列表示例
└── requirements.txt
```

## FAQ

### 1) 报错 `missing API key in env var: MOONSHOT_API_KEY`

未设置 `MOONSHOT_API_KEY`。请在 shell 环境或 `.env` 中配置后重试。

### 2) 报错 `output directory does not exist`

`translate-url-batch` 与原子写入都要求目标目录已存在。先执行 `mkdir -p <dir>`。

### 3) 报错 `no URLs found in: ...`

URL 列表文件为空，或全部是空行/注释行。请保证至少有一个有效 URL。

### 4) 报错 `markdown guardrails failed`

输出 Markdown 结构未通过最终护栏。建议先对输入做 `sanitize-md`，并降低复杂嵌套列表或异常 fence 结构。

### 5) 命令应该在 `src/` 下运行吗？

不需要。本文档所有命令均基于仓库根目录执行：`python -m translator ...`。

## License

当前仓库未提供明确的 `LICENSE` 文件。如需开源发布，建议先补充许可证文本。
