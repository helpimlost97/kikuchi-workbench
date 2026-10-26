#!/usr/bin/env python3
"""Clone or update the workbench's kikuchipy fork, check out the pinned commit and install it.

    python update_fork.py                      # clone (if needed), check out the pinned commit, install
    python update_fork.py --dir D:\\kikuchipy   # the clone lives elsewhere (default: kikuchipy next to this script)
    python update_fork.py --commit workbench   # another commit/branch (e.g. the newest workbench) instead of the pin
    python update_fork.py --no-install         # skip "pip install -e ."
    python update_fork.py --show-latest        # only list Johan's develop commits newer than the fork's base and exit

The fork is https://github.com/helpimlost97/kikuchipy, a GitHub fork of Johan Westraadt's
https://github.com/jwestraadt/kikuchipy. Its branch ``workbench`` is Johan's develop at BASE_COMMIT
plus the workbench's own commits (fast lazy .up1 loading, VRAM release after GPU runs, and newer
experimental work on top), so nothing has to be patched after the checkout.

What it does, in order:

1. Clones the fork into --dir when that folder does not exist, otherwise fetches its ``workbench``
   branch (by URL, so a clone whose ``origin`` is still Johan's repository works too).
2. Makes sure the working tree is clean. A tree whose files already equal the target commit (e.g. a
   clone of Johan's repository with the old kikuchipy_local_patches.patch applied) is reset, since
   nothing is lost; any other local change stops the script, so nothing of yours is lost.
3. Checks out the pinned commit (PINNED_COMMIT below, --commit overrides) as a detached HEAD.
4. Runs "pip install -e <dir>" with the Python that runs this script (unless --no-install) and prints
   the kikuchipy version and file that Python now imports.

CuPy is not touched; see README.md section 4 for the CuPy wheel.
"""
import argparse
import subprocess
import sys
from pathlib import Path

FORK_URL = "https://github.com/helpimlost97/kikuchipy.git"
UPSTREAM_URL = "https://github.com/jwestraadt/kikuchipy.git"
BRANCH = "workbench"
# The commit of the fork's workbench branch that everything in this folder was last run against: Johan's
# develop at BASE_COMMIT plus the two workbench patches. Update it together with README.md.
PINNED_COMMIT = "31d2882d"
BASE_COMMIT = "4c7a0035"
# Where step 1 fetches the workbench branch to, independent of how the clone's remotes are named.
BRANCH_REF = f"refs/remotes/workbench-fork/{BRANCH}"


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
    """Parse the command line and perform the four steps of the module docstring.

    Side effects: network access (git clone or fetch), changes to the working tree and HEAD of the clone, and
    unless --no-install a ``pip install -e`` into the environment of the running Python. Exits with a message
    through ``sys.exit`` as soon as a step cannot be done safely (foreign local changes, failing command).
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent / "kikuchipy"),
                    help="folder of the clone (default: kikuchipy next to this script)")
    ap.add_argument("--commit", default=PINNED_COMMIT, help=f"commit or branch to check out (default: {PINNED_COMMIT})")
    ap.add_argument("--no-install", action="store_true", help="do not run pip install -e")
    ap.add_argument("--show-latest", action="store_true",
                    help=f"list Johan's develop commits newer than {BASE_COMMIT} and exit")
    a = ap.parse_args()

    clone = Path(a.dir).expanduser().resolve()

    # Step 1: clone or fetch
    if not (clone / ".git").exists():
        if clone.exists() and any(clone.iterdir()):
            sys.exit(f"{clone} exists and is not a git clone; move it away or pass --dir")
        run(["git", "clone", FORK_URL, clone])
    run(["git", "fetch", FORK_URL, f"+refs/heads/{BRANCH}:{BRANCH_REF}"], cwd=clone)

    latest = short_hash(clone, BRANCH_REF)
    print(f"newest commit of the fork's {BRANCH} branch: {latest}")
    if a.show_latest:
        run(["git", "fetch", UPSTREAM_URL, "develop"], cwd=clone)
        run(["git", "log", "--oneline", f"{BASE_COMMIT}..FETCH_HEAD"], cwd=clone)
        return

    # Step 2: a clean tree. "workbench" means the fetched branch, not a stale local branch of that name.
    target = a.commit
    if target in (BRANCH, f"origin/{BRANCH}"):
        target = BRANCH_REF
    dirty = local_changes(clone)
    if dirty:
        same = run(["git", "diff", "--quiet", target], cwd=clone, check=False)
        if same.returncode != 0:
            print(dirty)
            sys.exit(f"{clone} has local changes; commit, stash or discard them first (git stash / git checkout -- .)")
        # Every modified file already holds the content of the target commit (e.g. the old patch applied
        # to Johan's commit), so discarding the modifications loses nothing.
        run(["git", "checkout", "--", "."], cwd=clone)
        print(f"discarded local changes that equal {a.commit} (the old applied patch)")

    # Step 3: the pinned commit
    run(["git", "-c", "advice.detachedHead=false", "checkout", "--detach", target], cwd=clone)
    head = short_hash(clone, "HEAD")
    print(f"checked out {head}")

    # Step 4: install. The import check runs from the home folder so that it cannot pick up a kikuchipy
    # folder in the current directory instead of the installed package.
    if not a.no_install:
        run([sys.executable, "-m", "pip", "install", "-e", clone])
        code = "import kikuchipy as kp; print('kikuchipy', kp.__version__, 'from', kp.__file__)"
        run([sys.executable, "-c", code], cwd=Path.home())
    print(f"\nDone: fork at {head}" + ("" if head == latest[: len(head)] else f" (the {BRANCH} branch is at {latest})"))


if __name__ == "__main__":
    main()
