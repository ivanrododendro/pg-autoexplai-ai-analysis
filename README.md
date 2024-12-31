# PostgreSQL Auto Explain Log AI Analysis Tool

This tool analyzes PostgreSQL auto_explain log files using AI (OpenAI GPT or Google Gemini) to provide optimization recommendations and generate interactive HTML reports.

## Features

- Analyzes PostgreSQL execution plans from auto_explain logs
- Leverages AI (GPT-4/3.5 or Gemini) to provide optimization recommendations
- Generates interactive HTML reports with:
    - Query execution plans visualization using pev2
    - AI-powered optimization suggestions
    - Query occurrence statistics
    - Overall analysis summary
- Supports token limit management for different AI models
- Handles multiple queries in a single log file
- Provides optimization recommendations

## Requirements

### Python Dependencies

```bash
pip install requests tiktoken google-generativeai asyncio logging
```

### API Keys

You need to set up API keys for either:
- OpenAI GPT (for gpt-4, gpt-3.5-turbo models)
- Google Gemini (for gemini-2.0-flash-exp and other Gemini models)

### PostgreSQL Configuration

Your PostgreSQL instance must be configured with auto_explain to generate execution plans in the log files.
Setting must be logged in order for the script to identify the end of th plan.

## Usage

### Basic Usage

```bash
python analyze_pg_logs.py path/to/postgresql.log
```

### Advanced Options

```bash
python analyze_pg_logs.py path/to/postgresql.log -m MODEL_NAME -c MAX_AI_CALLS
```

#### Parameters

- `log_filename`: Path to the PostgreSQL log file containing execution plans
- `-m, --model`: AI model to use for analysis (default: gemini-2.0-flash-exp)
    - Supported models:
        - OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
        - Gemini: gemini-2.0-flash-exp, gemini-1.5-flash, gemini-1.5-pro
- `-c, --max-ai-calls`: Maximum number of AI API calls to make (default: -1 for unlimited)

### Example

```bash
python analyze_pg_logs.py postgresql.log -m gpt-4o -c 10
```

This will:
1. Process the first 10 execution plans from postgresql.log
2. Use GPT-4o for analysis
3. Generate an HTML report at postgresql.log_report.html

## Output

The tool generates an HTML report containing:
- Individual query analysis with:
    - Execution timestamp
    - Query name/description
    - AI-generated optimization recommendations
    - Interactive execution plan visualization
- Summary section with:
    - Query occurrence statistics
    - Most common optimization patterns
    - Overall recommendations

## Notes

- The tool expects log entries to contain both query text and execution plans
- AI analysis is provided in French
- Token limits are enforced based on the selected model
- API keys should be properly configured before running the tool
