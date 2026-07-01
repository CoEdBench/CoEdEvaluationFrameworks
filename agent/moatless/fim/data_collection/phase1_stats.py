#!/usr/bin/env python3
"""
benchmark_stats.py
──────────────────
Automated evaluation script: reads JSONL format Benchmark dataset,
computes multi-dimensional feature distributions, outputs readable report and optionally saves JSON.

Usage:
    python benchmark_stats.py --input data.jsonl
    python benchmark_stats.py --input data.jsonl --output report.json
"""

import json
import re
import argparse
import statistics
from collections import Counter
from typing import List, Dict, Any, Optional


# ══════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r'\w+|[^\w\s]', text)

def count_tokens(text: str) -> int:
    return len(simple_tokenize(text))

def safe_mean(lst):
    return round(statistics.mean(lst), 3) if lst else 0.0

def safe_median(lst):
    return round(statistics.median(lst), 3) if lst else 0.0

def safe_stdev(lst):
    return round(statistics.stdev(lst), 3) if len(lst) >= 2 else 0.0

def percentile(lst, p):
    if not lst:
        return 0.0
    s = sorted(lst)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 3)

def distribution_buckets(values, buckets):
    """Bucket the numerical list by buckets boundaries, return count per bucket"""
    result = {}
    edges = list(buckets)
    for i, edge in enumerate(edges):
        lo = edges[i - 1] if i > 0 else float('-inf')
        hi = edge
        label = f"<={hi}" if i == 0 else f"{lo+1}~{hi}"
        result[label] = sum(1 for v in values if lo < v <= hi)
    result[f">{edges[-1]}"] = sum(1 for v in values if v > edges[-1])
    return result


# ══════════════════════════════════════════════════════════════════
# Diff Parsing
# ══════════════════════════════════════════════════════════════════

def parse_diff(diff_text: str) -> List[Dict]:
    """Parse unified diff, return a list of structured dicts for each hunk"""
    hunks = []
    hunk_re = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
    cur = None
    for line in diff_text.splitlines():
        m = hunk_re.match(line)
        if m:
            if cur:
                hunks.append(cur)
            cur = dict(
                old_start=int(m.group(1)),
                old_count=int(m.group(2) or 1),
                new_start=int(m.group(3)),
                new_count=int(m.group(4) or 1),
                added=[], deleted=[], context=[]
            )
        elif cur is not None:
            if line.startswith('+') and not line.startswith('+++'):
                cur['added'].append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                cur['deleted'].append(line[1:])
            elif line.startswith(' '):
                cur['context'].append(line[1:])
    if cur:
        hunks.append(cur)
    return hunks

def avg_indent(lines: List[str]) -> float:
    if not lines:
        return 0.0
    levels = [(len(l) - len(l.lstrip(' '))) / 4 for l in lines]
    return round(sum(levels) / len(levels), 2)

def extract_api_calls(lines: List[str]) -> List[str]:
    calls = set()
    for line in lines:
        calls.update(re.findall(r'(\w+)\s*\(', line))
    return list(calls)

def has_pattern(lines: List[str], pattern: str) -> bool:
    return bool(re.search(pattern, '\n'.join(lines)))


# ══════════════════════════════════════════════════════════════════
# Single Record Feature Extraction
# ══════════════════════════════════════════════════════════════════

def extract_record_features(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all statistical features from a single benchmark record"""
    feat = {}

    # ── 1. Commit Meta Info ────────────────────────────────────────
    feat['repo_name']         = record.get('repo_name', '')
    feat['commit_hash']       = record.get('hash', '')[:12]
    feat['is_merge']          = record.get('is_merge', False)
    feat['issue_count']       = len(record.get('issue_ids', []))
    feat['has_issue_ref']     = feat['issue_count'] > 0
    msg = record.get('msg', '')
    feat['commit_msg_length'] = len(msg)
    feat['commit_msg_tokens'] = count_tokens(msg)

    # ── 2. Source File Info ────────────────────────────────────────
    src_changes = record.get('source_changes', [])
    feat['source_file_count'] = len(src_changes)

    src_total_lines_list  = []
    src_total_tokens_list = []
    all_hunks_meta        = []   # Feature dict for each hunk

    for sc in src_changes:
        src_code    = sc.get('source_code', '')
        file_lines  = src_code.count('\n') + 1 if src_code else 0
        file_tokens = count_tokens(src_code)
        src_total_lines_list.append(file_lines)
        src_total_tokens_list.append(file_tokens)

        # ── 3. Hunk Level Statistics ───────────────────────────────
        hunks = parse_diff(sc.get('diff', ''))
        for h in hunks:
            added_text   = '\n'.join(h['added'])
            deleted_text = '\n'.join(h['deleted'])

            fill_lines  = len(h['added'])
            if fill_lines>50:
                pass
            if fill_lines ==0:
                continue
            fill_chars  = len(added_text)
            fill_tokens = count_tokens(added_text)
            del_lines   = len(h['deleted'])
            del_tokens  = count_tokens(deleted_text)
            ctx_lines   = len(h['context'])

            fill_apis = extract_api_calls(h['added'])
            del_apis  = extract_api_calls(h['deleted'])

            hunk_meta = dict(
                # Position info
                new_start      = h['new_start'],
                position_ratio = round(h['new_start'] / file_lines, 4) if file_lines else 0,
                # Fill target statistics
                fill_lines     = fill_lines,
                fill_chars     = fill_chars,
                fill_tokens    = fill_tokens,
                fill_avg_line_len = round(fill_chars / fill_lines, 2) if fill_lines else 0,
                fill_indent_lvl   = avg_indent(h['added']),
                # Deleted line statistics
                del_lines      = del_lines,
                del_tokens     = del_tokens,
                # Net change
                net_loc        = fill_lines - del_lines,
                # Context
                ctx_lines      = ctx_lines,
                # Content feature markers
                has_condition  = has_pattern(h['added'], r'\b(if|elif|else|not)\b'),
                has_func_def   = has_pattern(h['added'], r'\bdef\s+\w+'),
                has_import     = has_pattern(h['added'], r'\bimport\b'),
                has_comment    = has_pattern(h['added'], r'^\s*#'),
                # API change analysis
                api_introduced = list(set(fill_apis) - set(del_apis)),
                api_replaced   = list(set(del_apis)  - set(fill_apis)),
                api_overlap    = list(set(fill_apis) & set(del_apis)),
                fill_api_count = len(fill_apis),
                del_api_count  = len(del_apis),
            )
            all_hunks_meta.append(hunk_meta)

    feat['hunk_count'] = len(all_hunks_meta)

    # Summarize fill targets
    feat['total_fill_lines']  = sum(h['fill_lines']  for h in all_hunks_meta)
    feat['total_fill_chars']  = sum(h['fill_chars']  for h in all_hunks_meta)
    feat['total_fill_tokens'] = sum(h['fill_tokens'] for h in all_hunks_meta)
    feat['total_del_lines']   = sum(h['del_lines']   for h in all_hunks_meta)
    feat['total_del_tokens']  = sum(h['del_tokens']  for h in all_hunks_meta)
    feat['net_loc_change']    = sum(h['net_loc']     for h in all_hunks_meta)
    feat['total_ctx_lines']   = sum(h['ctx_lines']   for h in all_hunks_meta)

    # Fill content feature summary
    feat['fill_has_condition'] = any(h['has_condition'] for h in all_hunks_meta)
    feat['fill_has_func_def']  = any(h['has_func_def']  for h in all_hunks_meta)
    feat['fill_has_import']    = any(h['has_import']    for h in all_hunks_meta)
    feat['fill_has_comment']   = any(h['has_comment']   for h in all_hunks_meta)

    # API change summary
    all_introduced = list({a for h in all_hunks_meta for a in h['api_introduced']})
    all_replaced   = list({a for h in all_hunks_meta for a in h['api_replaced']})
    feat['api_introduced_count'] = len(all_introduced)
    feat['api_replaced_count']   = len(all_replaced)
    feat['api_change_type'] = (
        'pure_add'     if all_introduced and not all_replaced else
        'pure_replace' if all_replaced   and not all_introduced else
        'mixed'        if all_introduced and all_replaced else
        'refactor'
    )

    # Hunk position (use first hunk as representative)
    feat['hunk_position_ratio'] = all_hunks_meta[0]['position_ratio'] if all_hunks_meta else 0.0
    feat['fill_indent_level']   = all_hunks_meta[0]['fill_indent_lvl'] if all_hunks_meta else 0.0

    # Source file summary
    feat['src_file_total_lines']  = sum(src_total_lines_list)
    feat['src_file_total_tokens'] = sum(src_total_tokens_list)

    # ── 4. Test File Statistics ─────────────────────────────────────
    test_changes = record.get('test_changes', [])
    feat['test_file_count'] = len(test_changes)

    test_added_lines_total = 0
    test_del_lines_total   = 0
    new_test_funcs_all     = []
    test_file_tokens_total = 0
    test_file_lines_total  = 0

    for tc in test_changes:
        diff = tc.get('diff', '')
        src  = tc.get('source_code', '')
        added_lines   = [l[1:] for l in diff.splitlines()
                         if l.startswith('+') and not l.startswith('+++')]
        deleted_lines = [l[1:] for l in diff.splitlines()
                         if l.startswith('-') and not l.startswith('---')]
        test_added_lines_total += len(added_lines)
        test_del_lines_total   += len(deleted_lines)
        # New test functions (starting with +def test_ in diff)
        new_funcs = re.findall(r'^\+def\s+(test_\w+)', diff, re.MULTILINE)
        new_test_funcs_all.extend(new_funcs)
        test_file_lines_total  += src.count('\n') + 1 if src else 0
        test_file_tokens_total += count_tokens(src)

    feat['test_added_lines']        = test_added_lines_total
    feat['test_del_lines']          = test_del_lines_total
    feat['new_test_count']          = len(new_test_funcs_all)
    feat['new_test_functions']      = new_test_funcs_all
    feat['test_file_total_lines']   = test_file_lines_total
    feat['test_file_total_tokens']  = test_file_tokens_total

    # ── 5. Context Info (TODO) ─────────────────────────────────────
    # TODO: Need to access other repo files, not yet implemented
    # feat['cross_file_deps']       = TODO  # Cross-file dependency count
    # feat['import_graph_depth']    = TODO  # Dependency graph depth
    # feat['context_window_tokens'] = TODO  # Actual prompt context token count

    # ── 6. Comprehensive Difficulty Assessment ─────────────────────
    score  = 0
    score += min(feat['total_fill_lines'], 10)           # Line count, max 10 points
    score += min(feat['total_fill_tokens'] // 10, 10)    # Token count, max 10 points
    score += 5 if feat['fill_has_condition']    else 0   # Has conditional logic
    score += 5 if feat['api_introduced_count'] > 0 else 0  # Introduces new API
    score += 3 if feat['fill_has_import']       else 0   # Has import
    feat['difficulty_score'] = score
    feat['difficulty_level'] = (
        'easy'   if score <= 10 else
        'medium' if score <= 20 else
        'hard'
    )

    return feat


# ══════════════════════════════════════════════════════════════════
# Aggregate Statistics
# ══════════════════════════════════════════════════════════════════

def aggregate_stats(all_features: List[Dict]) -> Dict:
    """Aggregate statistics over all record features"""
    n = len(all_features)
    if n == 0:
        return {'total_records': 0}

    def collect(key):
        return [f[key] for f in all_features if key in f]

    def num_stats(key):
        vals = collect(key)
        return {
            'mean':   safe_mean(vals),
            'median': safe_median(vals),
            'stdev':  safe_stdev(vals),
            'min':    min(vals) if vals else 0,
            'max':    max(vals) if vals else 0,
            'p25':    percentile(vals, 25),
            'p75':    percentile(vals, 75),
            'p90':    percentile(vals, 90),
        }

    def bool_rate(key):
        vals = collect(key)
        true_cnt = sum(1 for v in vals if v)
        return {'count': true_cnt, 'rate': round(true_cnt / n, 4) if n else 0}

    def counter_dist(key):
        return dict(Counter(collect(key)).most_common())

    report = {'total_records': n}

    # 1. Repo distribution
    report['repos'] = counter_dist('repo_name')

    # 2. Commit meta info
    report['commit_info'] = {
        'is_merge':      bool_rate('is_merge'),
        'has_issue_ref': bool_rate('has_issue_ref'),
        'issue_count':   num_stats('issue_count'),
        'msg_length':    num_stats('commit_msg_length'),
        'msg_tokens':    num_stats('commit_msg_tokens'),
    }

    # 3. Source File
    report['source_file'] = {
        'total_lines':  num_stats('src_file_total_lines'),
        'total_tokens': num_stats('src_file_total_tokens'),
        'file_count':   num_stats('source_file_count'),
    }

    # 4. Hunk Statistics
    report['hunk'] = {
        'hunk_count':     num_stats('hunk_count'),
        'position_ratio': num_stats('hunk_position_ratio'),
        'position_distribution': distribution_buckets(
            collect('hunk_position_ratio'),
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        'indent_level': num_stats('fill_indent_level'),
        'ctx_lines':    num_stats('total_ctx_lines'),
    }

    # 5. Fill Target (Core)
    fill_lines_vals  = collect('total_fill_lines')
    fill_tokens_vals = collect('total_fill_tokens')
    report['fill_target'] = {
        'fill_lines': {
            **num_stats('total_fill_lines'),
            'distribution': distribution_buckets(
                fill_lines_vals, [1, 2, 3, 5, 8, 10, 15, 20]
            )
        },
        'fill_tokens': {
            **num_stats('total_fill_tokens'),
            'distribution': distribution_buckets(
                fill_tokens_vals, [10, 20, 30, 50, 80, 100, 150, 200]
            )
        },
        'fill_chars':     num_stats('total_fill_chars'),
        'del_lines':      num_stats('total_del_lines'),
        'del_tokens':     num_stats('total_del_tokens'),
        'net_loc_change': num_stats('net_loc_change'),
        'net_loc_distribution': counter_dist('net_loc_change'),
    }

    # 6. Completion Content Features
    report['fill_features'] = {
        'has_condition':   bool_rate('fill_has_condition'),
        'has_func_def':    bool_rate('fill_has_func_def'),
        'has_import':      bool_rate('fill_has_import'),
        'has_comment':     bool_rate('fill_has_comment'),
        'api_change_type': counter_dist('api_change_type'),
        'api_introduced':  num_stats('api_introduced_count'),
        'api_replaced':    num_stats('api_replaced_count'),
    }

    # 7. Test file
    report['test_file'] = {
        'test_file_count':        num_stats('test_file_count'),
        'test_added_lines':       num_stats('test_added_lines'),
        'test_del_lines':         num_stats('test_del_lines'),
        'new_test_count':         num_stats('new_test_count'),
        'test_file_total_lines':  num_stats('test_file_total_lines'),
        'test_file_total_tokens': num_stats('test_file_total_tokens'),
        'new_test_distribution':  distribution_buckets(
            collect('new_test_count'), [0, 1, 2, 3, 5]
        ),
    }

    # 8. Difficulty Distribution
    report['difficulty'] = {
        'score_stats':        num_stats('difficulty_score'),
        'level_distribution': counter_dist('difficulty_level'),
    }

    # 9. Context Info (TODO)
    report['context_info'] = {
        '_note': 'TODO: Need to access other repo files, not yet implemented',
        # 'cross_file_deps':       TODO,
        # 'import_graph_depth':    TODO,
        # 'context_window_tokens': TODO,
    }

    return report


# ══════════════════════════════════════════════════════════════════
# Report Rendering
# ══════════════════════════════════════════════════════════════════

def render_num(d: Dict) -> str:
    return (f"mean={d['mean']}, median={d['median']}, stdev={d['stdev']}, "
            f"min={d['min']}, max={d['max']}, "
            f"p25={d['p25']}, p75={d['p75']}, p90={d['p90']}")

def render_bool(d: Dict) -> str:
    return f"{d['count']} items ({d['rate']*100:.1f}%)"

def bar_chart(cnt, total, width=20) -> str:
    return '█' * int(cnt / total * width) if cnt and total else ''

def print_report(report: Dict):
    n = report['total_records']
    SEP = "═" * 64

    print(f"\n{SEP}")
    print(f"  📊  Benchmark Dataset Statistics Report (Total {n} records)")
    print(SEP)

    # 1. Repository Distribution
    print("\n▌ 1. Repository Distribution")
    for repo, cnt in report['repos'].items():
        print(f"    {repo:<30} {cnt:>5} items ({cnt/n*100:.1f}%)")

    # 2. Commit Metadata
    ci = report['commit_info']
    print("\n▌ 2. Commit Meta Info")
    print(f"    Merge Commit:       {render_bool(ci['is_merge'])}")
    print(f"    Has Issue Ref:      {render_bool(ci['has_issue_ref'])}")
    print(f"    Issue Count:         {render_num(ci['issue_count'])}")
    print(f"    Commit Msg Length:  {render_num(ci['msg_length'])}")
    print(f"    Commit Msg Tokens:  {render_num(ci['msg_tokens'])}")

    # 3. Source File
    sf = report['source_file']
    print("\n▌ 3. Source File Info")
    print(f"    File Lines:         {render_num(sf['total_lines'])}")
    print(f"    File Tokens:        {render_num(sf['total_tokens'])}")
    print(f"    files modified:         {render_num(sf['file_count'])}")

    # 4. Hunk
    hk = report['hunk']
    print("\n▌ 4. Hunk Statistics")
    print(f"    hunk count:          {render_num(hk['hunk_count'])}")
    print(f"    Hunk Position:      {render_num(hk['position_ratio'])}")
    print(f"    Indent Level:       {render_num(hk['indent_level'])}")
    print(f"    Context Lines:      {render_num(hk['ctx_lines'])}")
    print(f"    Hunk Position Distribution:")
    max_cnt = max(hk['position_distribution'].values(), default=1)
    for bucket, cnt in hk['position_distribution'].items():
        print(f"      {bucket:<12} {cnt:>4} items {bar_chart(cnt, max_cnt)}")

    # 5. Fill Target
    ft = report['fill_target']
    print("\n▌ 5. Fill Target Statistics (Core)")
    print(f"    Fill Lines:         {render_num(ft['fill_lines'])}")
    print(f"    Fill Tokens:        {render_num(ft['fill_tokens'])}")
    print(f"    Fill Chars:         {render_num(ft['fill_chars'])}")
    print(f"    lines removed:           {render_num(ft['del_lines'])}")
    print(f"    Deleted Tokens:     {render_num(ft['del_tokens'])}")
    print(f"    Net LOC Change:     {render_num(ft['net_loc_change'])}")
    print(f"\n    Fill Lines Distribution:")
    max_cnt = max(ft['fill_lines']['distribution'].values(), default=1)
    for bucket, cnt in ft['fill_lines']['distribution'].items():
        print(f"      {bucket:<12} {cnt:>4} items {bar_chart(cnt, max_cnt)}")
    print(f"\n    Fill Tokens Distribution:")
    max_cnt = max(ft['fill_tokens']['distribution'].values(), default=1)
    for bucket, cnt in ft['fill_tokens']['distribution'].items():
        print(f"      {bucket:<12} {cnt:>4} items {bar_chart(cnt, max_cnt)}")
    print(f"\n    Net LOC Change Distribution (Top 10):")
    for delta, cnt in list(ft['net_loc_distribution'].items())[:10]:
        print(f"      {str(delta):<8} {cnt:>4} items")

    # 6. Completion Content Features
    ff = report['fill_features']
    print("\n▌ 6. Completion Content Features")
    print(f"    Has Condition:      {render_bool(ff['has_condition'])}")
    print(f"    Has Func Def:       {render_bool(ff['has_func_def'])}")
    print(f"    Has Import:         {render_bool(ff['has_import'])}")
    print(f"    Has Comment:        {render_bool(ff['has_comment'])}")
    print(f"    API Introduced:     {render_num(ff['api_introduced'])}")
    print(f"    API Replaced:       {render_num(ff['api_replaced'])}")
    print(f"    API Change Type Distribution:")
    for t, cnt in ff['api_change_type'].items():
        print(f"      {t:<20} {cnt:>4} items ({cnt/n*100:.1f}%)")

    # 7. Test file
    tf = report['test_file']
    print("\n▌ 7. Test File Statistics")
    print(f"    Test File Count:    {render_num(tf['test_file_count'])}")
    print(f"    Test File Total Lines: {render_num(tf['test_file_total_lines'])}")
    print(f"    Test File Tokens:   {render_num(tf['test_file_total_tokens'])}")
    print(f"    Test Lines Added:   {render_num(tf['test_added_lines'])}")
    print(f"    Test Lines Removed: {render_num(tf['test_del_lines'])}")
    print(f"    New Test Functions: {render_num(tf['new_test_count'])}")
    print(f"    New Test Functions Distribution:")
    max_cnt = max(tf['new_test_distribution'].values(), default=1)
    for bucket, cnt in tf['new_test_distribution'].items():
        print(f"      {bucket:<12} {cnt:>4} items {bar_chart(cnt, max_cnt)}")

    # 8. Difficulty Distribution
    df = report['difficulty']
    print("\n▌ 8. Difficulty Assessment")
    print(f"    Difficulty Score:   {render_num(df['score_stats'])}")
    print(f"    Difficulty Level Distribution:")
    for lvl in ['easy', 'medium', 'hard']:
        cnt = df['level_distribution'].get(lvl, 0)
        print(f"      {lvl:<10} {cnt:>4} items ({cnt/n*100:.1f}%)  {bar_chart(cnt, n, 30)}")

    # 9. Context (TODO)
    print("\n▌ 9. Context Info Statistics")
    print(f"    ⚠️  {report['context_info']['_note']}")
    print(f"    To Be Implemented:")
    print(f"      - cross_file_deps:       Cross-file dependency count")
    print(f"      - import_graph_depth:    Dependency graph depth")
    print(f"      - context_window_tokens: Actual prompt context token count")

    print(f"\n{SEP}\n")


# ══════════════════════════════════════════════════════════════════
# Main Entry
# ══════════════════════════════════════════════════════════════════

def analyze_jsonl(filepath: str, output_json: Optional[str] = None):
    """
    Read JSONL file, extract features, output statistics report.

    Args:
        filepath:    Input JSONL file path
        output_json: Optional, save aggregate statistics results to JSON file
    Returns:
        (report dict, all_features list)
    """
    all_features = []
    errors = []

    print(f"📂 Reading: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                feat = extract_record_features(record)
                all_features.append(feat)
            except Exception as e:
                errors.append({'line': line_no, 'error': str(e)})

    print(f"✅ Successfully parsed {len(all_features)} items, failed {len(errors)} items")
    if errors:
        print("⚠️  Failed lines (top 5):")
        for e in errors[:5]:
            print(f"   Line {e['line']}: {e['error']}")

    report = aggregate_stats(all_features)
    print_report(report)

    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"💾 Statistics saved to: {output_json}")

    return report, all_features


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='Benchmark dataset multi-dimensional statistical analysis tool'
    # )
    # parser.add_argument(
    #     '--input', '-i', required=True,
    #     help='Input JSONL file path'
    # )
    # parser.add_argument(
    #     '--output', '-o', default=None,
    #     help='Optional: save aggregate statistics results to JSON file'
    # )
    # args = parser.parse_args()
    input = r'D:\Data\2025\xty\data_collection\input.jsonl'
    output = input+".stats"
    analyze_jsonl(input, output)
