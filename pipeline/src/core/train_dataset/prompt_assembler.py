"""
prompt_assembler.py
===================
Assembles Stage1 / Stage2 messages list.
Keeps identical alignment with assembler.py used during inference.
Training and inference share the same prompt templates.
"""

import difflib
import re
from typing import List, Optional

from src.domain.types import ParsedItem


# ══════════════════════════════════════════════════════════════════════════
# Diff Utilities
# ══════════════════════════════════════════════════════════════════════════

def build_unified_diff(
        before: str,
        after: str,
        file_path: str,
        from_lineno: int = 1,
        to_lineno: int = 1,
) -> str:
    """Generate unified diff with @@ line numbers aligned to actual file positions."""
    before_lines = before.splitlines(keepends=True)
    after_lines  = after.splitlines(keepends=True)

    raw = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
        lineterm="",
    ))
    if not raw:
        return "(no diff — before and after are identical)"

    hunk_re     = re.compile(r'^(@@ -)(\d+)(,\d+)?( \+)(\d+)(,\d+)?( @@.*)')
    from_offset = from_lineno - 1
    to_offset   = to_lineno   - 1
    result      = []

    for line in raw:
        m = hunk_re.match(line)
        if m:
            line = (
                f"@@ -{int(m.group(2)) + from_offset}{m.group(3) or ''}"
                f" +{int(m.group(5)) + to_offset}{m.group(6) or ''}"
                f"{m.group(7)}"
            )
        result.append(line)

    return "".join(result)


# ══════════════════════════════════════════════════════════════════════════
# File extension -> Markdown code block language tag
# ══════════════════════════════════════════════════════════════════════════

_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".java": "java", ".go": "go",
    ".rb": "ruby", ".rs": "rust", ".cpp": "cpp", ".c": "c",
    ".cs": "csharp", ".php": "php", ".kt": "kotlin", ".swift": "swift",
    ".sh": "bash", ".html": "html", ".css": "css",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".sql": "sql",
}


def _get_lang_tag(file_path: str) -> str:
    ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return _EXT_TO_LANG.get(ext, "")


# ══════════════════════════════════════════════════════════════════════════
# PromptAssembler
# ══════════════════════════════════════════════════════════════════════════

class PromptAssembler:
    """Assembles Stage1 / Stage2 messages list. Keeps alignment with assembler.py used during inference."""

    # ──────────────────────────────────────────────────────────────
    # Stage1: Identify impacted locations
    # ──────────────────────────────────────────────────────────────

    def stage1(self, parsed_item: ParsedItem, context_report: dict) -> List[dict]:
        root_hunk = parsed_item.root_hunk
        root_diff = build_unified_diff(
            before=parsed_item.root_before_code,
            after=parsed_item.root_after_code,
            file_path=root_hunk.file_path,
            from_lineno=root_hunk.old_start_line,
            to_lineno=root_hunk.start_line,
        )
        return [
            {"role": "system", "content": _STAGE1_SYSTEM},
            {"role": "user",   "content": self._stage1_user(parsed_item, root_diff, context_report)},
        ]

    def _stage1_user(self, parsed_item: ParsedItem, root_diff: str, context_report: dict) -> str:
        root_hunk  = parsed_item.root_hunk
        ctx_blocks = []
        for ctx in context_report.get("related_contexts", []):
            ctx_blocks.append(
                f"### {ctx['file_path']} (lines {ctx.get('usage_line_nums', [])})\n"
                f"Reason: {ctx.get('reason', '')}\n"
                f"```\n{ctx.get('relevant_code', '')}\n```"
            )
        context_str = "\n\n".join(ctx_blocks) if ctx_blocks else "(none)"

        return f"""\
## Change Requirement
{getattr(parsed_item, 'requirement', '(not provided)')}

## Root Change (Already Applied)
File: `{root_hunk.file_path}`
```diff
{root_diff}
```

## Related Code Snippets
The following code snippets show the current state of related code
(already reflecting the root change above). Determine if they still
need additional updates for consistency.
{context_str}

## Task
Analyze the root change and identify ALL other locations that need updates.

**How to analyze (chain-of-thought):**

1. **Classify the root change.** What kind of change is it?
   - Adding/removing items from a type/enum/union?
   - Renaming a function/type/variable?
   - Adding/removing/changing a function signature?
   - Adding/removing a method call?

2. **Identify the dependency rule.** If X changed, what other constructs
   depend on X and would need the same treatment? Examples:
   - Enum members added → every switch/case or mapping that enumerates them needs updating
   - Method added to interface → every implementation needs the new method
   - Type field added → all usages referencing that field may need updating
   - Function renamed → all call sites need the new name

3. **Enumerate all affected locations.** Apply the dependency rule from
   step 2 to the code in Related Code Snippets. There may be 0, 1, or
   MULTIPLE impacted locations — list ALL of them.

4. **Verify each location.** For each location you identify:
   - Is this genuinely a NEW location that needs changes (not the root
     change location itself, which has already been handled)?
   - Does the specific line number match what you see in the snippet?

**Output requirements:**
- Do NOT include the root change's own location as an impacted location
  (it's already been applied).
- For each impacted location, specify the exact line numbers that need
  changes (refer to the line numbers shown in the code snippets).
- In the `reason` field, explain WHAT specific change is needed on those
  lines and WHY the dependency rule applies there.
- Set `"impacted_locations": []` if no additional locations need updating.

Respond with JSON:
```json
{{
  "reasoning": "<Describe the dependency rule derived from the root change. E.g.: 'Root change adds X and Y to enum — all functions that map enum values to behavior must handle the new values.'>",
  "impacted_locations": [
    {{
      "file": "<relative file path>",
      "lines": [<specific line numbers needing changes>],
      "reason": "<Explain what specific change is needed and why the dependency rule applies here.>"
    }}
  ]
}}
```"""

    # ──────────────────────────────────────────────────────────────
    # Stage2: Generate next_version
    # ──────────────────────────────────────────────────────────────

    def stage2(
            self,
            parsed_item: ParsedItem,
            file_path: str,
            current_version: str,
            reasoning: str,
            impacted_lines: Optional[List[int]] = None,
            location_reason: str = "",
            snippet_start_line: int = 1,
    ) -> List[dict]:
        """
        Build Stage 2 messages.
        Keeps alignment with assemble_fix() used during inference.

        Args:
            parsed_item: Contains Root change info
            file_path: Target file relative path
            current_version: Code snippet extracted by pipeline at line number +/-10
            reasoning: Overall reasoning from Stage 1 output
            impacted_lines: Stage 1 identified impacted line numbers for this file
            location_reason: Stage 1 reason for this file
            snippet_start_line: Start line of current snippet in file
        """
        root_hunk = parsed_item.root_hunk

        root_diff = build_unified_diff(
            before=parsed_item.root_before_code,
            after=parsed_item.root_after_code,
            file_path=root_hunk.file_path,
            from_lineno=root_hunk.old_start_line,
            to_lineno=root_hunk.start_line,
        )

        return [
            {"role": "system", "content": _STAGE2_SYSTEM},
            {"role": "user",   "content": self._stage2_user(
                parsed_item, root_diff, file_path, current_version,
                reasoning, impacted_lines, location_reason, snippet_start_line,
            )},
        ]

    def _stage2_user(
            self,
            parsed_item: ParsedItem,
            root_diff: str,
            file_path: str,
            current_version: str,
            reasoning: str,
            impacted_lines: Optional[List[int]] = None,
            location_reason: str = "",
            snippet_start_line: int = 1,
    ) -> str:
        root_hunk = parsed_item.root_hunk
        snippet_end_line = snippet_start_line + len(current_version.splitlines()) - 1

        lines_hint = ""
        if impacted_lines:
            lines_str = ", ".join(str(ln) for ln in impacted_lines)
            lines_hint = (
                f"\n### Stage 1 Identified Lines\n"
                f"These specific lines in this file need modification: [{lines_str}]\n"
                f"The code below spans file lines {snippet_start_line}~{snippet_end_line}.\n"
                f"Change ONLY these lines — do not modify any other part of the snippet.\n"
            )
        else:
            lines_hint = (
                f"\n### Stage 1 Identified Lines\n"
                f"The code below spans file lines {snippet_start_line}~{snippet_end_line}.\n"
            )

        analysis_hint = ""
        if reasoning:
            analysis_hint = f"\n### Stage 1 Analysis\nOverall impact: {reasoning}\n"
        if location_reason:
            analysis_hint += f"Changes needed in this file: {location_reason}\n"

        return f"""\
## Change Requirement
{getattr(parsed_item, 'requirement', '(not provided)')}

## Root Change Pattern
The following change was made in `{root_hunk.file_path}`. Apply the **same transformation pattern** to the target file below.
```diff
{root_diff}
```
{analysis_hint}
## Target File: `{file_path}`

### Current Code (file lines {snippet_start_line}~{snippet_end_line})
```
{current_version}
```
{lines_hint}
## Task
Update the target file's code snippet to be consistent with the root change.

**Instructions:**
- Change ONLY the lines listed in "Stage 1 Identified Lines" above.
- Make MINIMAL edits — change or replace individual lines, do not rewrite the entire snippet.
- If the root change renames a function/type/variable in the root file, apply the same rename to the corresponding lines in this target file.
- If none of the identified lines need changes, set "next_version" to null.
- Output the code WITHOUT line number prefixes — just clean source code.

Output JSON only:
```json
{{
  "next_version": "<updated full snippet, or null if no change needed>",
  "change_summary": "<one-line description of what was changed, or 'no change needed'>"
}}
```"""


# ══════════════════════════════════════════════════════════════════════════
# System Prompts (module-level constants, avoid repeated instantiation)
# ══════════════════════════════════════════════════════════════════════════

_STAGE1_SYSTEM = """\
You are an expert software engineer performing impact analysis.

You will receive:
1. A change requirement describing the developer's intent.
2. A diff of the root change ALREADY APPLIED to one file.
3. Code snippets from potentially related files in the codebase
   (the root change is already reflected in these snippets).

Your task: determine what ELSE needs to change in these related snippets
to stay consistent with the root change.

Focus on SEMANTIC DEPENDENCIES — constructs that reference, extend, or
are derived from the changed code. Do NOT flag the root change location
itself (it's already been handled).

Respond with valid JSON only."""

_STAGE2_SYSTEM = """\
You are an expert software engineer making precise, minimal edits.

Given a root change and a target code snippet with identified lines,
update ONLY the identified lines as needed.

Rules:
- Output the **entire** updated snippet (the full code block), not just changed lines.
- **Change ONLY the specific lines identified in "Stage 1 Identified Lines".**
- Do NOT reformat, reorganize, or restructure surrounding code.
- Preserve indentation, style, and surrounding logic exactly as-is.
- If none of the identified lines need changes, set "next_version" to null.
- CRITICAL: Respond with ONLY valid JSON. No markdown, no code fences (```), no explanation.
  Start with '{' and end with '}'."""
