import os
import shutil
import logging
import subprocess

import stat

logger = logging.getLogger(__name__)


def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _get_bare_path(source_repo: str, sandbox_root: str) -> str:
    """Each source_repo has only one bare repo, cached under bare/ next to sandbox_root."""
    bare_root = os.path.join(os.path.dirname(os.path.abspath(sandbox_root)), "bare")
    repo_name = os.path.basename(os.path.normpath(source_repo))
    return os.path.join(bare_root, f"{repo_name}.git")


def _git(args: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Wrapper around subprocess.run to call git."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        check=check,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def prepare_sandbox_repo(source_repo: str, sandbox_root: str, commit_hash: str) -> str:
    """
    Use bare clone + worktree instead of full clone.

    Each source_repo is bare-cloned only once (cached in bare/ directory),
    then `git worktree add` creates a lightweight working directory (~1s, no object copying).
    """
    target_path = os.path.abspath(sandbox_root)
    bare_path   = _get_bare_path(source_repo, sandbox_root)

    # ── 1. Ensure bare repo exists ────────────────────────────────────
    if not os.path.exists(bare_path):
        os.makedirs(os.path.dirname(bare_path), exist_ok=True)
        logger.info("bare clone %s → %s", source_repo, bare_path)
        _git(["clone", "--bare", source_repo, bare_path])

    # ── 2. Clean up existing worktree (leftover from last run) ───────
    if os.path.exists(target_path):
        logger.info("removing stale worktree: %s", target_path)
        _git(["worktree", "remove", "--force", target_path],
             cwd=target_path, check=False)

    # ── 3. Create worktree (checkout to commit^) ────────────────────
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    checkout_target = f"{commit_hash}^"
    logger.info("worktree add %s @ %s", target_path, checkout_target)
    _git(["worktree", "add", "--force", target_path, checkout_target],
         cwd=bare_path)

    return target_path


def cleanup_sandbox_repo(sandbox_path: str) -> bool:
    """
    Clean up worktree with `git worktree remove`, fallback to rmtree on failure.
    Avoids leftover `.git/worktrees/` metadata.
    """
    target = os.path.abspath(sandbox_path)
    if not os.path.exists(target):
        return False

    # ── Find the bare repo from the worktree's .git file ──────────
    git_file = os.path.join(target, ".git")
    if os.path.isfile(git_file):
        try:
            with open(git_file) as f:
                line = f.readline().strip()
            if line.startswith("gitdir:"):
                gitdir_path = line.split("gitdir: ", 1)[1].strip()
                # gitdir: .../bare/DefinitelyTyped.git/worktrees/<name>
                idx = gitdir_path.replace("\\", "/").find("/bare/")
                if idx >= 0:
                    bare_repo = gitdir_path[:gitdir_path.index("/worktrees/", idx)]
                    _git(["worktree", "remove", "--force", target],
                         cwd=bare_repo, check=False)
                    logger.info("worktree removed: %s", target)
                    return True
        except Exception:
            pass

    # Fallback: rmtree
    try:
        shutil.rmtree(target, onerror=remove_readonly)
        logger.info("sandbox cleaned (rmtree): %s", target)
        return True
    except Exception as e:
        logger.warning("failed to clean sandbox %s: %s", target, e)
        return False
