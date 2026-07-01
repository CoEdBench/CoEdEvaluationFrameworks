import re

from pydriller import Repository
from loguru import logger
from typing import List, Generator, Iterator, Optional

from moatless.fim.data_collection.config.settings import MiningConfig
from moatless.fim.data_collection.core.types import CommitCandidate, FileChange
from moatless.fim.data_collection.filters.base import BaseFilter


class RepoMiner:
    def __init__(self, repo_name:str,repo_path: str, filters: List[BaseFilter]):
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.filters = filters
        self.config = MiningConfig()

    def mine(self, limit: int ) -> Iterator[CommitCandidate]:
        repo = Repository(self.repo_path, order='reverse',only_no_merge=False)

        count = 0
        for commit in repo.traverse_commits():
            if commit.merge:
                # print(f"Commit {commit.hash} is a merge commit, usually has no modified_files {len(commit.modified_files)}")
                pass
            else:
                # print(f"Commit {commit.hash} file count: {len(commit.modified_files)}")
                pass
            # Run filters
            passed = True
            for f in self.filters:
                if not f.check(commit, self.config):
                    passed = False
                    break

            if not passed:
                continue

            # Extract data
            candidate = self._extract(commit)
            if candidate:
                yield candidate
                count += 1
                if count >= limit:
                    break

    def _extract(self, commit) -> Optional[CommitCandidate]:
        try:
            # [New] 1. Extract Issue ID
            # Common patterns: #123, gh-123, fix #123.
            # Use general regex to extract all # numbers
            issue_pattern = re.compile(r'#(\d+)')
            found_issues = issue_pattern.findall(commit.msg)
            # Deduplicate
            issue_ids = list(set(found_issues))


            source_changes = []
            test_changes = []

            for f in commit.modified_files:
                fname = f.filename.lower()

                # Determine type
                is_test = False
                for pattern in self.config.TEST_FILE_PATTERNS:
                    if pattern in fname:
                        is_test = True
                        break
                if commit.hash == '72d97bceec1af70d9b65bb49d12abdd5694e67f1':
                    pass
                # If it is a source code file, check if it is in the ignore list
                if any(fname.endswith(ext) for ext in self.config.SOURCE_EXTENSIONS):
                    is_ignored = False
                    for ignore_pattern in self.config.IGNORE_FILES:
                        if ignore_pattern in fname:
                            is_ignored = True
                            break
                    if is_ignored:
                        continue
                else:
                    continue

                # Build FileChange object
                change = FileChange(
                    old_path=f.old_path,
                    new_path=f.new_path,
                    change_type=f.change_type.name,
                    diff=f.diff,
                    source_code=f.source_code,
                    is_test=is_test
                )

                if is_test:
                    test_changes.append(change)
                else:
                    source_changes.append(change)

            # Double check: ensure non-empty condition is still satisfied after extraction
            if not source_changes or not test_changes:
                return None

            return CommitCandidate(
                repo_name=self.repo_name,
                hash=commit.hash,
                msg=commit.msg,
                author_date=str(commit.author_date),
                issue_ids=issue_ids,
                repo_url=self.repo_path,
                is_merge=commit.merge,
                source_changes=source_changes,
                test_changes=test_changes,
                source_files_count=len(source_changes),
                test_files_count=len(test_changes),
                metadata={"issue_details": {}}
            )
        except Exception as e:
            logger.error(f"Error extracting commit {commit.hash}: {e}")
            return None


import os
from github import Github
from typing import List, Dict


class IssueEnricher:
    def __init__(self, token: str, repo_slug: str):
        """
        :param token: GitHub Personal Access Token
        :param repo_slug: Format "owner/repo" (e.g., "psf/requests")
        """
        self.gh = Github(token)
        self.repo = self.gh.get_repo(repo_slug)
        # Simple cache to avoid repeated requests for the same Issue
        self.cache: Dict[str, dict] = {}

    def enrich(self, candidate: CommitCandidate) -> CommitCandidate:
        """
        Receive a Candidate, fill its Issue description information into metadata
        """
        if not candidate.issue_ids:
            return candidate

        issue_details = {}

        for issue_id in candidate.issue_ids:
            # Check cache
            if issue_id in self.cache:
                issue_details[issue_id] = self.cache[issue_id]
                continue

            try:
                # Call API to get Issue object
                # Note: This is a network request and may be slow; for production, consider async or batch queries
                gh_issue = self.repo.get_issue(int(issue_id))

                info = {
                    "title": gh_issue.title,
                    "body": gh_issue.body,  # This is the Issue description content you need
                    "state": gh_issue.state,
                    "labels": [l.name for l in gh_issue.labels]
                }

                # Store in cache and results
                self.cache[issue_id] = info
                issue_details[issue_id] = info

                logger.info(f"Fetched issue #{issue_id} for commit {candidate.hash[:7]}")

            except Exception as e:
                logger.warning(f"Failed to fetch issue #{issue_id}: {e}")
                issue_details[issue_id] = {"error": str(e)}

        # Store the fetched data into metadata
        candidate.metadata["issue_details"] = issue_details

        return candidate
