"""
LLM-powered Python Bug Detector
Analyzes Python source files for bugs using an LLM.
"""

import sys
import json
import argparse
from pathlib import Path
import urllib.request
import urllib.error


def read_source_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    if p.suffix != ".py":
        print(f"[WARNING] File does not have .py extension: {path}")
    return p.read_text(encoding="utf-8")


def build_prompt(code: str, filename: str) -> str:
    return f"""You are an expert Python code reviewer and debugger.
Analyze the following Python code for bugs, logic errors, and potential issues.

File: {filename}

```python
{code}
```

Respond ONLY with a JSON object in this exact format (no markdown, no explanation outside JSON):
{{
  "summary": "One sentence overall assessment",
  "bugs": [
    {{
      "line": <line_number_or_null>,
      "severity": "critical|warning|info",
      "description": "What is wrong",
      "fix": "How to fix it"
    }}
  ],
  "score": <integer_0_to_10_where_10_is_bug_free>
}}

If no bugs are found, return an empty list for "bugs" and score 10.
"""


def call_llm(prompt: str, api_key: str, model: str = "claude-opus-4-6") -> str:
    """Call Anthropic Claude API."""
    payload = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[ERROR] API request failed ({e.code}): {body}")
        sys.exit(1)


def parse_response(raw: str) -> dict:
    """Parse JSON from LLM response, stripping any accidental markdown."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"summary": "Could not parse LLM response.", "bugs": [], "score": -1, "raw": raw}


SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}


def print_report(result: dict, filename: str) -> None:
    print("\n" + "=" * 60)
    print(f"  Bug Report: {filename}")
    print("=" * 60)
    print(f"  Summary : {result.get('summary', 'N/A')}")
    score = result.get("score", -1)
    bar = "█" * score + "░" * (10 - score) if 0 <= score <= 10 else "N/A"
    print(f"  Score   : {score}/10  [{bar}]")
    print("-" * 60)

    bugs = result.get("bugs", [])
    if not bugs:
        print("  ✅ No bugs detected!")
    else:
        for i, bug in enumerate(bugs, 1):
            icon = SEVERITY_ICON.get(bug.get("severity", "info"), "⚪")
            line_info = f"Line {bug['line']}" if bug.get("line") else "General"
            print(f"\n  {i}. {icon} [{bug.get('severity','?').upper()}] — {line_info}")
            print(f"     Problem : {bug.get('description', '')}")
            print(f"     Fix     : {bug.get('fix', '')}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered Python bug detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python bug_detector.py my_script.py --api-key sk-ant-..."
    )
    parser.add_argument("file", help="Python source file to analyze")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Model to use (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")
    args = parser.parse_args()

    print(f"[*] Reading {args.file} ...")
    code = read_source_file(args.file)
    print(f"[*] Sending to LLM ({args.model}) ...")
    prompt = build_prompt(code, args.file)
    raw = call_llm(prompt, args.api_key, args.model)
    result = parse_response(raw)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result, args.file)


if __name__ == "__main__":
    main()