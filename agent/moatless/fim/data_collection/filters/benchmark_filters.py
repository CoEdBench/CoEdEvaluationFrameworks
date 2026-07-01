from .base import BaseFilter
from pydriller import Commit

from ..config.settings import MiningConfig

"""
Ignore DELETE/ADD operations	Only focus on MODIFY type changes
Test file detection	Match TEST_FILE_PATTERNS
Source code file detection	.py suffix + not test + not ignored
Must have test	has_test == True
Source file count limit	MIN_SOURCE_FILES ≤ count ≤ MAX_SOURCE_FILES
Source LOC change limit	MIN_SOURCE_LOC ≤ loc ≤ MAX_SOURCE_LOC
"""


def count_hunks(modified_file) -> int:
    """
    Calculate the number of consecutive modified blocks (Hunks) in a ModifiedFile.
    Merge the added and deleted line numbers, then count by consecutive grouping.
    """
    parsed = modified_file.diff_parsed
    added_lines = [lineno for lineno, _ in parsed.get("added", [])]
    deleted_lines = [lineno for lineno, _ in parsed.get("deleted", [])]

    all_lines = sorted(set(added_lines + deleted_lines))

    if not all_lines:
        return 0

    hunk_count = 1
    for i in range(1, len(all_lines)):
        if all_lines[i] - all_lines[i - 1] > 1:
            hunk_count += 1

    return hunk_count


class BenchmarkFilter(BaseFilter):
    """
    Benchmark-specific filter

    Logic:
    1. Must be a Merge Commit
    2. Must include at least one Source File modification (as Input/Target)
    3. Must include at least one Test File modification (as Verifier)
    4. Source Files size must be within the threshold range (Test Files size is unrestricted)
    """

    def check(self, commit: Commit, config: MiningConfig) -> bool:

        source_files_count = 0
        test_files_count = 0
        source_loc_change = 0
        source_hunk_count = 0
        has_test = False

        for f in commit.modified_files:
            if f.change_type.name == 'DELETE':
                return False
            if f.change_type.name == 'ADD':
                return False
            if len([lineno for lineno, _ in f.diff_parsed.get("added", [])]) == 0:
                return False

            fname = f.filename.lower()

            # Determine whether it is a test file
            is_test = False
            for pattern in config.TEST_FILE_PATTERNS:
                if pattern in fname:
                    is_test = True
                    has_test = True
                    test_files_count += 1
                    # break

            # Determine whether it is a valid source file
            if any(fname.endswith(ext) for ext in config.SOURCE_EXTENSIONS) and not is_test:
                is_ignored = False
                for ignore_pattern in config.IGNORE_FILES:
                    if ignore_pattern in fname:
                        is_ignored = True
                        break

                if not is_ignored:
                    source_files_count += 1
                    source_loc_change += (f.added_lines + f.deleted_lines)
                    source_hunk_count += count_hunks(f)
        # 2. Core check logic
        # print(f"Test files count: {test_files_count}; Source files count: {source_files_count}")
        # A. Must have tests
        if not has_test:
            return False
        # B. Must have source files with count meeting requirements
        if not (config.MIN_SOURCE_FILES <= source_files_count <= config.MAX_SOURCE_FILES):
            return False

        # C. Source modification size meets requirements
        if not (config.MIN_SOURCE_LOC <= source_loc_change <= config.MAX_SOURCE_LOC):
            return False
        # D. Hunk count must meet requirements
        if not (config.MIN_SOURCE_HUNKS <= source_hunk_count <= config.MAX_SOURCE_HUNKS):
            return False
        return True
