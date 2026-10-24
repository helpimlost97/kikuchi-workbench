#!/usr/bin/env python3
"""Clone or update Johan Westraadt's kikuchipy fork to the pinned commit and apply the local patches.

    python update_fork.py                      # clone (if needed), check out the pinned commit, patch, install
    python update_fork.py --dir D:\\kikuchipy   # the clone lives elsewhere (default: ./kikuchipy)
    python update_fork.py --commit develop     # another commit/branch (e.g. the newest develop) instead of the pin
    python update_fork.py --no-install         # skip "pip install -e ."
    python update_fork.py --show-latest        # only print the newest commit of origin/develop and exit

What it does, in order:

1. Clones https://github.com/jwestraadt/kikuchipy into --dir when that folder does not exist,
   otherwise fetches origin/develop in it.
2. Makes sure the working tree is clean. The only local change it knows how to undo is this
   project's own patch (kikuchipy_local_patches.patch): if that is what is applied, it is
   reverted first. Any other local change stops the script, so nothing of yours is lost.
3. Checks out the pinned commit (PINNED_COMMIT below, --commit overrides) as a detached HEAD.
4. Applies kikuchipy_local_patches.patch (git apply --check first, so a patch that does not fit a
   newer commit is reported instead of half applied).
5. Runs "pip install -e <dir>" with the Python that runs this script (unless --no-install) and prints
   the kikuchipy version and file that Python now imports.

CuPy is not touched; see README.md section 4 for the CuPy wheel.
"""
import argparse
import subprocess
import sys
from pathlib import Path

FORK_URL = "https://github.com/jwestraadt/kikuchipy.git"
# The commit of Johan's develop branch that everything in this folder was last run against.
# Update it together with README.md when moving to a newer commit.
PINNED_COMMIT = "4c7a0035"
PATCH = Path(__file__).resolve().parent / "kikuchipy_local_patches.patch"


def run(args, cwd=None, check=True, capture=False):
    """Run one command, echoing it first as a "$ ..." line so the output reads like a terminal session.

    Parameters
    ----------
    args : sequence
        The command and its arguments; every item is converted with ``str()``, so ``Path`` objects are fine.
    cwd : path-like or None
        Working directory of the command (None: the current one).
    check : bool
        When True, a non-zero exit code ends the script through ``sys.exit`` with a one-line message (the
        captured output, if any, is printed first). When False, the caller reads ``returncode`` itself.
    capture : bool
        When True, stdout and stderr are captured together into ``result.stdout`` instead of going to the
        terminal.

    Returns
    -------
    subprocess.CompletedProcess
        With ``stdout`` set only when ``capture`` is True.
    """
    print("$", " ".join(str(a) for a in args), flush=True)
    result = subprocess.run(
        [str(a) for a in args], cwd=cwd, check=False, text=True,
        stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None,
    )
    if check and result.returncode != 0:
        if capture:
            print(result.stdout)
        sys.exit(f"command failed with exit code {result.returncode}")
    return result


def short_hash(clone, ref):
    """The abbreviated commit hash of ``ref`` (a branch, tag or commit) in the clone at ``clone``."""
    return run(["git", "rev-parse", "--short", ref], cwd=clone, capture=True).stdout.strip()


def local_changes(clone):
    """The ``git status --porcelain`` listing of the modified tracked files of the clone; empty when it is clean.

    Untracked files are ignored on purpose: build output and editor files are harmless to a checkout.
    """
    return run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=clone, capture=True).stdout.strip()


def main():
    """Parse the command line and perform the five steps of the module docstring.

    Side effects: network access (git clone or fetch), changes to the working tree and HEAD of the clone, and
    unless --no-install a ``pip install -e`` into the environment of the running Python. Exits with a message
    through ``sys.exit`` as soon as a step cannot be done safely (missing patch file, foreign local changes,
    patch not fitting the checked-out commit, failing command).
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="kikuchipy", help="folder of the clone (default: ./kikuchipy)")
    ap.add_argument("--commit", default=PINNED_COMMIT, help=f"commit or branch to check out (default: {PINNED_COMMIT})")
    ap.add_argument("--no-install", action="store_true", help="do not run pip install -e")
    ap.add_argument("--show-latest", action="store_true", help="print the newest commit of origin/develop and exit")
    a = ap.parse_args()

    clone = Path(a.dir).expanduser().resolve()
    if not PATCH.exists():
        sys.exit(f"patch file not found: {PATCH}")

    # Step 1: clone or fetch
    if not (clone / ".git").exists():
        if clone.exists() and any(clone.iterdir()):
            sys.exit(f"{clone} exists and is not a git clone; move it away or pass --dir")
        run(["git", "clone", FORK_URL, clone])
    else:
        run(["git", "fetch", "origin", "develop"], cwd=clone)

    latest = short_hash(clone, "origin/develop")
    print(f"newest commit of origin/develop: {latest}")
    if a.show_latest:
        run(["git", "log", "--oneline", f"{PINNED_COMMIT}..origin/develop"], cwd=clone)
        return

    # Step 2: a clean tree, undoing only our own patch
    dirty = local_changes(clone)
    if dirty:
        reverse_ok = run(["git", "apply", "--reverse", "--check", PATCH], cwd=clone, check=False, capture=True)
        if reverse_ok.returncode != 0:
            print(dirty)
            sys.exit(
                f"{clone} has local changes which are not (only) this project's patch; "
                "commit, stash or discard them first (git stash / git checkout -- .)"
            )
        run(["git", "apply", "--reverse", PATCH], cwd=clone)
        print("reverted the previously applied local patch")
        # The reverse-apply check only looks at the files the patch touches: anything else that is still
        # modified is someone's own work and must not be carried through the checkout.
        dirty = local_changes(clone)
        if dirty:
            print(dirty)
            sys.exit(f"{clone} has local changes besides this project's patch; commit, stash or discard them first")

    # Step 3: the pinned commit. "develop" means the fetched branch, not a stale local branch of that name.
    target = a.commit
    if target in ("develop", "origin/develop"):
        target = "origin/develop"
    run(["git", "-c", "advice.detachedHead=false", "checkout", "--detach", target], cwd=clone)
    head = short_hash(clone, "HEAD")
    print(f"checked out {head}")

    # Step 4: the patch
    check = run(["git", "apply", "--check", PATCH], cwd=clone, check=False, capture=True)
    if check.returncode != 0:
        print(check.stdout)
        sys.exit(f"kikuchipy_local_patches.patch does not apply to {head}; the fork changed the patched files. "
                 "Check out the pinned commit (run without --commit) or rebase the patch.")
    run(["git", "apply", PATCH], cwd=clone)
    changed = run(["git", "status", "--short"], cwd=clone, capture=True).stdout.strip()
    print("patched files:\n" + changed)

    # Step 5: install. The import check runs from the home folder so that it cannot pick up a kikuchipy
    # folder in the current directory instead of the installed package.
    if not a.no_install:
        run([sys.executable, "-m", "pip", "install", "-e", clone])
        code = "import kikuchipy as kp; print('kikuchipy', kp.__version__, 'from', kp.__file__)"
        run([sys.executable, "-c", code], cwd=Path.home())
    print(f"\nDone: fork at {head}" + ("" if head == latest[: len(head)] else f" (origin/develop is at {latest})"))


if __name__ == "__main__":
    main()
