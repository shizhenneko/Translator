# Course Notes AI Study-Guide + Translator

A Python CLI tool that transforms English course notes into polished Chinese study guides with AI-powered translation, annotations, and strict preservation of math/code/Markdown structures.

## Features

- **Dual Input**: URL (via Jina Reader) or local Markdown files
- **Smart Translation**: Global outline + glossary extraction, then chunk-by-chunk translation
- **Preservation-First**: Protects math formulas, code blocks, links, and Markdown structure
- **Medium Annotations**: Key explanations with examples/analogies
- **Atomic Output**: Fail-fast behavior prevents partial/corrupted files
- **Concurrent Processing**: 2-4 workers for faster translation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export MOONSHOT_API_KEY="your-kimi-api-key"

# Optional: For URL fetching
export JINA_API_KEY="your-jina-api-key"

# Or use a .env file
# MOONSHOT_API_KEY=your-kimi-api-key
# JINA_API_KEY=your-jina-api-key

The CLI loads `.env` automatically on all platforms before reading env vars.
```

## Usage

### Translate Local Markdown

```bash
cd src
python -m translator translate-md \
  --in ../path/to/input.md \
  --out ../path/to/output.md \
  --max-chunk-chars 8000 \
  --concurrency 3
```

### Translate from URL

```bash
cd src
python -m translator translate-url \
  --url https://cs231n.github.io/neural-networks-1/ \
  --out ../translated_note.md
```

### Debug Commands

```bash
# Test placeholder protection
python -m translator debug-protect \
  --in input.md \
  --out protected.md \
  --map map.json

# Test restoration
python -m translator debug-restore \
  --in protected.md \
  --map map.json \
  --out restored.md

# Test chunking
python -m translator debug-chunk \
  --in input.md \
  --max-chunk-chars 8000 \
  --json

# Test URL fetching
python -m translator debug-fetch \
  --url https://example.com \
  --out fetched.md
```

## Output Structure

The generated Markdown includes:

1. **Meta Section**: Source, timestamp, model info
2. **Global Outline**: Hierarchical structure with key takeaways
3. **Glossary**: Term translations with notes
4. **Translated Body**: Chunk-by-chunk translation with annotations

## Architecture

### Map-Reduce Pipeline

1. **Step 0**: Fetch/read input (Jina Reader or local file)
2. **Step 1**: Global profiling (outline + glossary extraction via JSON mode)
3. **Chunking**: Split by headings/paragraphs (8000 char limit)
4. **Step 2**: Translate chunks with concurrency (protect -> translate -> restore -> QA)
5. **Assembly**: Combine meta + outline + glossary + body
6. **Output**: Atomic write (temp + rename)

### Preservation Layer

Protected elements:
- Fenced code blocks: ` ```...``` `
- Inline code: `` `...` ``
- Math: `$...$`, `$$...$$`, `\\(...\\)`, `\\[...\\]`, `\\begin{...}\\end{...}`
- Links/images: `[text](URL)`, `![alt](URL)`
- HTML tags, tables, footnotes

## Testing

```bash
cd src
python -m pytest ../tests -q

# Results: 33 passed, 3 skipped (integration tests require MOONSHOT_API_KEY)
```

## Project Structure

```
translator/
├── src/translator/          # Main package
│   ├── cli.py              # CLI entry point
│   ├── pipeline.py         # End-to-end orchestration
│   ├── preservation.py     # Placeholder protection
│   ├── chunking.py         # Markdown-aware splitting
│   ├── llm_client.py       # Kimi API wrapper
│   ├── step1_profile.py    # Global profiling
│   ├── step2_translate.py  # Chunk translation
│   └── jina_reader_fetcher.py  # URL fetching
├── tests/                  # Test suite
│   ├── test_preservation.py
│   ├── test_chunking.py
│   └── test_integration.py
├── tests/fixtures/         # Test data
│   ├── sample.md
│   ├── protection.md
│   └── large.md
└── requirements.txt        # Dependencies
```

## Configuration

### LLM Settings

- **Model**: `kimi-k2-0905-preview`
- **Base URL**: `https://api.moonshot.cn/v1`
- **API Key**: `MOONSHOT_API_KEY` environment variable
- **Temperature**: 0.2-0.4 (configurable)
- **Timeout**: 60s (configurable)

### Translation Settings

- **Chunk Size**: 8000 chars (default, configurable via `--max-chunk-chars`)
- **Concurrency**: 3 workers (default, configurable via `--concurrency`)
- **Annotation Density**: Medium (key explanation + 1 example)
- **Term Style**: First occurrence `中文（English）`, then Chinese only
- **Glossary Enforcement**: Soft (prefer, warn in QA)

## QA Gates (Hard Fail)

- Placeholder multiset match (all placeholders present exactly once)
- No leftover placeholders after restoration
- Code fence count preserved
- Math delimiter counts preserved
- URLs unchanged
- UTF-8 output with required sections in correct order

## Known Issues

- Minor trailing newline issue in chunker reconstruction (non-critical)
- Python 3.8 compatibility requires `typing.Match` instead of `re.Match[str]`

## Development

Built with:
- Python 3.8+
- OpenAI Python SDK (for Kimi API)
- requests (for Jina Reader)
- tenacity (for retry logic)
- pytest (for testing)

## License

See project documentation for license details.
