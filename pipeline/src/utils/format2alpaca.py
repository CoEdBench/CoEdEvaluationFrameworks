import json
import hashlib
import argparse
from pathlib import Path


def make_hash(*fields: str) -> str:
    """Hash multiple fields with MD5 for deduplication."""
    combined = "\x00".join(fields)  # Use invisible separator to avoid concatenation ambiguity
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def convert_jsonl_to_training_format(
    input_path: str,
    output_path: str,
    dedup_mode: str = "full",  # "full" | "instruction_system" | "instruction"
):
    """
    """Convert JSONL data to training format JSON with deduplication."""

    Input format:
        - messages[0]: system prompt
        - messages[1]: user instruction
        - ground_truth: model response

    Output format:
        [{"instruction": ..., "input": "", "output": ..., "system": ...}]

    Deduplication modes:
        - "full"               : Deduplicate only when instruction + system + output are identical (default, strictest)
        - "instruction_system" : Deduplicate when instruction + system match
        - "instruction"        : Deduplicate when instruction matches only (most aggressive)
    """
    results = []
    seen_hashes: set[str] = set()

    skipped_parse = 0
    skipped_dedup = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num} JSON parse failed, skipped: {e}")
                skipped_parse += 1
                continue

            messages = item.get("messages", [])

            # Extract content by role
            system_content = ""
            user_content = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system_content = content
                elif role == "user":
                    user_content = content

            # Wrap ground_truth as output with code block
            ground_truth = item.get("ground_truth", "")
            output_content = "```json\n" + ground_truth + "\n```\n"

            # ---- Deduplication ----
            if dedup_mode == "full":
                key = make_hash(user_content, system_content, output_content)
            elif dedup_mode == "instruction_system":
                key = make_hash(user_content, system_content)
            elif dedup_mode == "instruction":
                key = make_hash(user_content)
            else:
                raise ValueError(f"Unknown dedup_mode: {dedup_mode!r}, "
                                 f"valid options: 'full' / 'instruction_system' / 'instruction'")

            if key in seen_hashes:
                skipped_dedup += 1
                continue
            seen_hashes.add(key)
            # ---- Deduplication end ----

            converted = {
                "instruction": user_content,
                "input": "",
                "output": output_content,
                "system": system_content,
            }
            results.append(converted)

    # Write output JSON file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Conversion complete!")
    print(f"   Input file   : {input_path}")
    print(f"   Output file  : {output_path}")
    print(f"   Dedup mode   : {dedup_mode}")
    print(f"   Converted    : {len(results)} records")
    print(f"   Dedup dropped: {skipped_dedup} records")
    if skipped_parse:
        print(f"   Parse failed : {skipped_parse} records")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert JSONL to training format")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--dedup-mode", default="full",
                        choices=["full", "instruction_system", "instruction"],
                        help="Deduplication mode (default: full)")
    args = parser.parse_args()
    convert_jsonl_to_training_format(args.input, args.output, dedup_mode=args.dedup_mode)