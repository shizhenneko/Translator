# 课程笔记 AI 导读与翻译器（translator）

一个面向英文课程笔记和文档页面的 Python CLI 工具：
输入网页 URL 或本地 Markdown，输出适合直接阅读和在 VS Code Markdown Preview Enhanced 中预览的中文 Markdown，并尽量保持公式、代码块、链接与标题结构稳定。

## 功能概览

- 双输入模式：支持 URL 抓取（Jina Reader）与本地 Markdown 文件
- 双模式输出：默认 `readable` 直接产出适合阅读的正文；`analysis` 保留 Meta / Outline / Glossary
- 结构保护优先：保护代码块、行内代码、数学公式、链接等敏感片段
- Markdown 护栏：内置 sanitize + normalize + autofix + lint，降低渲染风险
- 可读性优先：默认输出忠实、清爽，只有在确实有助理解时才补充少量说明
- 原子写入：写入失败不污染目标文件
- 可并发翻译：支持多线程分块并行
- 单命令工作流：默认只需一条 `translate` 命令
- Snapdown 处理：URL 模式下可将 `snapdown` 图块自动转换为 `mermaid`

## 快速开始

### 1. 环境要求

- Python 3.8+
- 可用的 DeepSeek API Key（必需）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量（推荐使用 `.env`）

在仓库根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
# 可选
JINA_API_KEY=your_jina_api_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

CLI 启动时会自动加载 `.env`。

### 4. 最短可运行示例

#### 示例 A：从 URL 直接翻译

```bash
python -m translator translate \
  --url https://sp21.datastructur.es/materials/proj/proj2/proj2 \
  --out output/proj2.zh.md
```

#### 示例 B：翻译本地 Markdown

```bash
python -m translator translate \
  --in documents/6.031_note1.md \
  --out output/6.031_note1.zh.md
```

说明：

- `output/` 不需要预先创建，程序会自动建目录
- 默认输出格式是 `readable`，适合直接阅读和渲染
- 只有在需要分析信息时才使用 `--output-format analysis`

## 命令总览

统一入口：

```bash
python -m translator --help
```

### 主要命令

| 命令 | 用途 | 关键参数 |
| --- | --- | --- |
| `translate` | 统一入口，翻译单个 URL / 本地 Markdown / URL 列表 | `--url` 或 `--in` 或 `--url-list` |
| `translate-md` | 翻译本地 Markdown | `--in`, `--out` |
| `translate-url` | 抓取并翻译单个 URL | `--url`, `--out` |
| `translate-url-batch` | 批量翻译 URL 列表 | `--url-list`, `--out-dir` |
| `lint-md` | 检查 Markdown 结构风险 | `--in` |
| `sanitize-md` | 预清洗 Markdown 抓取噪音 | `--in`, (`--out` 或 `--in-place`) |

### 常用翻译参数（`translate-*`）

- `--max-chunk-chars`：分块上限，默认 `5000`
- `--concurrency`：并发数，CLI 默认 `2`
- `--prompt-outline-mode`：`headings`（默认，提示词更短）或 `full`
- `--prompt-glossary-mode`：`filtered`（默认，仅注入相关术语）或 `full`
- `--output-format`：`readable`（默认，适合直接阅读）或 `analysis`
- `--timeout`：URL 抓取超时秒数，默认 `30.0`

### URL 模式参数

- `--no-snapdown-mermaid`：关闭 Snapdown -> Mermaid 自动转换
- `--jina-api-key-env`：从指定环境变量读取 Jina Key，并注入为 `JINA_API_KEY`

### 批量 URL 示例

```bash
python -m translator translate \
  --url https://example.com/a \
  --url https://example.com/b \
  --out-dir output/batch
```

说明：

- 重复传 `--url` 即可直接批量，不需要先准备列表文件
- `url.txt` 每行一个 URL
- 空行与 `#` 开头行会被忽略
- `--out-dir` 会自动创建
- 输出文件名会按序号和 URL slug 自动生成，例如 `001-sp21-datastructur-es-materials-proj-proj2-proj2.md`

如果你已经有 URL 列表文件，也可以继续使用：

```bash
python -m translator translate \
  --url-list url.txt \
  --out-dir output/batch
```

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

默认 `readable` 输出包含：

1. 一级标题
2. `Source: ...` 来源行
3. 正文翻译内容

如果使用 `--output-format analysis`，会额外包含：

1. `## Meta`
2. `## Outline`
3. `## Glossary`

## 配置说明

### LLM 相关环境变量

| 变量名 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | 无 | DeepSeek API Key |
| `DEEPSEEK_MODEL` | 否 | `deepseek-chat` | 模型名覆盖，默认对应 DeepSeek-V3.2 |
| `DEEPSEEK_BASE_URL` | 否 | `https://api.deepseek.com` | 官方接口地址 |
| `JINA_API_KEY` | 否 | 无 | URL 抓取鉴权（按需） |

### 运行时调优环境变量

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `TRANSLATOR_RETRY_LOG` | `1` | 重试日志开关，`0` 关闭 |
| `TRANSLATOR_STRICT_RENDERER` | `true` | 严格 markdown-it 渲染安全检查 |
| `TRANSLATOR_MAX_SAFE_LIST_DEPTH` | `1` | 列表中代码块安全深度上限 |
| `TRANSLATOR_GLOSSARY_MAX_TERMS` | `30` | 每块注入术语条目上限 |
| `TRANSLATOR_GLOSSARY_MAX_CHARS` | `2000` | 每块术语注入字符预算 |
| `TRANSLATOR_TIMING_LOG` | `1` | 输出关键阶段耗时日志，`0` 关闭 |

## 架构概览（Map-Reduce 风格）

1. 读取输入：URL 抓取或本地文件加载
2. 输入清洗：修正常见抓取残留与 Markdown 异常
3. 文档画像：`readable` 使用轻量本地画像，`analysis` 使用完整 Step1 画像
4. 分块：按标题/段落感知切块
5. Step2 分块翻译：占位保护 -> 翻译 -> 还原 -> QA
6. 拼装输出：`readable` 输出标题 + 来源 + 正文；`analysis` 再附加分析区块
7. 护栏校验：autofix + lint，不安全则失败
8. 落盘：原子写入，避免部分写入损坏

## 测试

运行全部测试：

```bash
pytest -q
```

说明：

- 集成测试会在缺少 `DEEPSEEK_API_KEY` 时自动跳过
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
├── translator/               # 根目录兼容包，仅保留入口与桥接
├── tests/                    # 测试用例
├── documents/                # 示例/产物文档
├── url.txt                   # URL 列表示例
└── requirements.txt
```

## FAQ

### 1) 报错 `missing API key in env var: DEEPSEEK_API_KEY`

未设置 `DEEPSEEK_API_KEY`。请在 shell 环境或 `.env` 中配置后重试。

### 2) 输出目录不存在怎么办？

不需要手动创建。`translate` / `translate-url` / `translate-url-batch` 都会自动创建目标目录。

### 3) 报错 `no URLs found in: ...`

URL 列表文件为空，或全部是空行/注释行。请保证至少有一个有效 URL。

### 4) 报错 `markdown guardrails failed`

输出 Markdown 结构未通过最终护栏，程序会直接失败，避免写出无法正常渲染的文档。可先运行 `lint-md` 检查已有输出，或调小 `--max-chunk-chars` 重新翻译。

### 5) 命令应该在 `src/` 下运行吗？

不需要。本文档所有命令均基于仓库根目录执行：`python -m translator ...`。

## License

当前仓库未提供明确的 `LICENSE` 文件。如需开源发布，建议先补充许可证文本。
