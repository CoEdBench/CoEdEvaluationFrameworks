"""
fim/prompt.py
Prompt construction: system prompt / user prompt / code context extraction
"""
from pathlib import Path
from typing import Any

from moatless.fim.schema import FillRequest

MULTI_HUNK_BLOCK_START = "<<<TARGET_HUNK>>>"
MULTI_HUNK_BLOCK_END = "<<<END_TARGET_HUNK>>>"


def build_context_snippet(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    context_lines: int,
    use_fill_marker: bool = True,
) -> str:
    """Extract code context from a file. Inserts <FILL_IN> marker at the target range by default."""
    full_path = Path(repo_path) / file_path
    with open(full_path, encoding="utf-8") as f:
        lines = f.readlines()

    begin = max(0, start_line - 1 - context_lines)
    end = min(len(lines), end_line + context_lines)

    if not use_fill_marker:
        return "".join(lines[begin:end])

    before = "".join(lines[begin : start_line - 1])
    after = "".join(lines[end_line:end])
    marker = "<FILL_IN>\n"
    return f"{before}{marker}{after}"


def build_system_prompt(max_iterations: int) -> str:
    """Build the system prompt, incorporating chain-of-thought reasoning, structured workflow, and budget limits."""
    max_explore = max(0, max_iterations - 1)
    return f"""\
You are an autonomous AI assistant with superior programming skills. \
You are working autonomously and cannot communicate with the user — \
rely solely on the information you can obtain from the available tools.

# Chain-of-Thought Reasoning
- Before starting any work — and whenever you encounter complex reasoning \
or decision points — use the **Think** tool to log your chain-of-thought.
- Always call Think by passing your reasoning as a plain string.
- Think at the very beginning of the task, and again whenever you need to \
plan the next step or resolve ambiguity.
- The chain-of-thought is internal; do not expose it in your final output.

# Workflow

1. **Understand the Task**
   - Read the task description carefully.
   - Identify which file(s) and line range(s) need to be changed.
   - Determine what additional context is required (dependencies, callers, tests).

2. **Locate Code**
   Use the following tools to find relevant code:
   - **GrepTool** — search across the codebase using regex patterns (preferred for finding where functions/variables are defined or referenced).
   - **ListFiles** — list files in a directory to understand project structure.
   - **ViewCode** — read a specific file or line range (use when you know the exact file and location).
   - **ReadFile** — read an entire file when full context is needed.

3. **Modify Code**
   Apply changes with:
   - **StringReplace** — replace an exact text string in a file with new content.

4. **Iterate as Needed**
   - Repeat steps 2–3 until all required changes are complete.
   - Once you have gathered sufficient context to understand what to change, \
you MUST immediately switch to **StringReplace** — do not read more files \
than necessary.
   - Stop as soon as you have enough information — do not exhaust the budget unnecessarily.

5. **Complete the Task**
   - Call **Finish** immediately after your last StringReplace.
   - Explain why the task is complete.

# Important Guidelines

- **Focus on the specific task** — implement only what is required; \
do not modify unrelated code.
- **Never guess** — do not assume line numbers or code content; \
always read the file to verify before editing.
- **Switch from reading to editing** — once you understand what needs to change, \
stop reading files and immediately apply **StringReplace**. Reading more files \
after you already know what to change wastes budget and will not help.
- **State management** — keep track of files you have already read and \
changes you have already applied; do not re-fetch the same data or \
repeat previous steps.
- **Call Finish last** — do not make any exploration calls after your \
final StringReplace.
- **When in doubt** — read the target file first before searching elsewhere.\
"""

def _get_metadata(task: FillRequest) -> dict[str, Any]:
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def _extract_commit_hash(task: FillRequest) -> str | None:
    metadata = _get_metadata(task)
    for key in ("commit_hash", "hash", "target_commit"):
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return None


def _extract_commit_message(task: FillRequest) -> str | None:
    metadata = _get_metadata(task)
    for key in ("commit_message", "commit_msg", "msg"):
        value = metadata.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    return None


def _extract_requirement_summary(task: FillRequest) -> str | None:
    metadata = _get_metadata(task)
    for key in ("requirement_summary", "issue_description", "requirement", "summary","re_msg"):
        value = metadata.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    return None


def _should_hide_commit_context(task: FillRequest) -> bool:
    metadata = _get_metadata(task)
    value = metadata.get("hide_commit_context")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _extract_task_mode(task: FillRequest) -> str | None:
    metadata = _get_metadata(task)
    mode = metadata.get("task_mode")
    if mode is None:
        return None
    text = str(mode).strip()
    return text or None


def _extract_known_input_hunk(task: FillRequest) -> dict[str, Any]:
    metadata = _get_metadata(task)
    hunk = metadata.get("known_input_hunk")
    if isinstance(hunk, dict):
        return hunk
    return {}


def _build_multi_hunk_instruction_block(task: FillRequest) -> str:
    known_hunk = _extract_known_input_hunk(task)
    known_hunk_added_text = str((_get_metadata(task)).get("known_input_hunk_added_text") or "").rstrip("\n")

    known_hunk_file = str(known_hunk.get("file_path") or task.file_path)
    known_hunk_start = known_hunk.get("start_line", task.start_line)
    known_hunk_end = known_hunk.get("end_line", task.end_line)

    edit_guidance = (
        "Editing instructions:\n"
        "1) The requirement summary describes what needs to be changed.\n"
        "2) The known input hunk (below) is one of the changes — it has already been applied.\n"
        "3) Explore the repository to find ALL OTHER locations that need changes.\n"
        "4) As soon as you understand what needs to be changed at a location, "
        "use StringReplace to apply the edit immediately — do not keep reading files.\n"
        "5) Do NOT revert or re-edit the known input hunk unless strictly necessary.\n"
        "6) After all edits are done, call Finish.\n"
    )

    known_input_block = (
        "Known input hunk (index=0, already applied):\n"
        f"- file_path={known_hunk_file} start_line={known_hunk_start} end_line={known_hunk_end}\n"
        "Added lines in this hunk:\n"
        f"{known_hunk_added_text}\n"
    )

    return (
        "Task mode: multi-hunk completion.\n"
        "The requirement spans multiple locations in the codebase. "
        "One change (the 'known input hunk') has already been applied as a starting point. "
        "You must discover and apply the remaining changes yourself.\n\n"
        f"{known_input_block}\n"
        f"{edit_guidance}"
    )


def build_user_prompt(task: FillRequest) -> str:
    """Build the user prompt, including repository information and code context."""
    task_mode = _extract_task_mode(task)
    use_fill_marker = task_mode != "phase3_multi_hunk_one_shot"
    context = build_context_snippet(
        repo_path=task.repo_path,
        file_path=task.file_path,
        start_line=task.start_line,
        end_line=task.end_line,
        context_lines=task.context_lines,
        use_fill_marker=use_fill_marker,
    )

    show_commit_context = not _should_hide_commit_context(task)
    commit_hash = _extract_commit_hash(task) if show_commit_context else None
    commit_message = _extract_commit_message(task) if show_commit_context else None
    requirement_summary = _extract_requirement_summary(task)

    commit_context_block = ""
    if commit_hash:
        commit_context_block += f"Commit hash: {commit_hash}\n"
    if commit_message:
        commit_context_block += f"Commit message:\n{commit_message}\n"
    if commit_context_block:
        commit_context_block += "\n"

    requirement_block = ""
    if requirement_summary:
        requirement_block = f"Requirement summary:\n{requirement_summary}\n\n"

    extra_mode_block = ""
    single_point_block = ""
    range_line = ""

    if task_mode == "phase3_multi_hunk_one_shot":
        extra_mode_block = _build_multi_hunk_instruction_block(task) + "\n"
        range_line = f"Known input hunk range: lines {task.start_line}-{task.end_line}\n\n"
    else:
        single_point_block = (
            f"Task: The `<FILL_IN>` marker in the code below marks where code needs to be written. "
            f"Your goal is to determine the correct implementation and use StringReplace to write it.\n\n"
            f"Workflow:\n"
            f"1) Understand the requirement — read the commit message and requirement summary (if provided).\n"
            f"2) Gather context — read the target file and any related files (tests, dependencies) as needed.\n"
            f"3) Write the code — use StringReplace to replace the `<FILL_IN>` marker with the correct implementation.\n"
            f"4) Call Finish — once StringReplace is done, call Finish immediately.\n\n"
            f"Constraints:\n"
            "- Use StringReplace to replace the `<FILL_IN>` marker line with your implementation.\n"
            "- The old_str should be exactly `<FILL_IN>` (or the marker line as it appears in the file).\n"
            "- Do not modify any other lines.\n"
        )

    return (
        f"{single_point_block}"
        f"{extra_mode_block}"
        f"Target file: {task.file_path}\n"
        f"{range_line}"
        f"{requirement_block}"
        f"Context:\n```{task.language}\n{context}\n```\n\n"
    )
