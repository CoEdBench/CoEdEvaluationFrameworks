from .base import BaseFilter
from pydriller import Commit
import os

from ..config.settings import MiningConfig


class MergeCommitFilter(BaseFilter):
    """Exclude Merge Commit"""

    def check(self, commit: Commit, config: MiningConfig) -> bool:
        return not commit.merge


class RelevantCodeFilter(BaseFilter):
    """
    Core filter: determines whether a Commit is worth keeping.
    Logic:
    1. Must include at least one .py file modification.
    2. The file must not be in the ignore list (IGNORE_FILES).
    3. If REQUIRE_TEST_CHANGE is True, it must also include a test file.
    """

    def check(self, commit: Commit, config: MiningConfig) -> bool:
        has_valid_source = False
        has_test = False

        for f in commit.modified_files:
            fname = f.filename.lower()

            # 1. Check whether it is a test
            is_current_file_test = False
            for pattern in config.TEST_FILE_PATTERNS:
                if pattern in fname:
                    has_test = True
                    is_current_file_test = True
                    break

            # 2. Check whether it is valid source code
            # Conditions: ends with .py + not a test file + not a blacklisted file
            if fname.endswith('.py') and not is_current_file_test:
                is_ignored = False
                for ignore_pattern in config.IGNORE_FILES:
                    if ignore_pattern in fname:
                        is_ignored = True
                        break

                if not is_ignored:
                    has_valid_source = True

        # Decision logic
        if config.REQUIRE_TEST_CHANGE:
            return has_valid_source and has_test
        else:
            # Lenient mode: as long as there is a valid source code modification
            # Note: we usually do not want to only capture test file modifications (that is test refactoring, not code evolution)
            # So here we enforce has_valid_source
            return has_valid_source


class SizeFilter(BaseFilter):
    """
    Filter by modification size.
    Note: only count the size of .py files, ignore documentation and other miscellaneous items.
    """

    def check(self, commit: Commit, config: MiningConfig) -> bool:
        # Filter all Python files (including tests)
        py_files = [
            f for f in commit.modified_files
            if f.filename.endswith('.py')
        ]

        # Check file count
        if not (config.MIN_MODIFIED_FILES <= len(py_files) <= config.MAX_MODIFIED_FILES):
            return False

        # Check lines of code changes (only count Python files)
        total_change = 0
        for f in py_files:
            total_change += (f.added_lines + f.deleted_lines)

        if not (config.MIN_LOC_CHANGE <= total_change <= config.MAX_LOC_CHANGE):
            return False

        return True