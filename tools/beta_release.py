"""Prepare the Git integration ref and the Windows beta archive."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile


BETA_FILES = {
    "anyxboxx.exe": "target/release/anyxboxx.exe",
    "anyxboxx-launcher.exe": "target/release/anyxboxx-launcher.exe",
    "victory.exe": "third_party/victory/target/release/victory.exe",
    "LICENSE": "LICENSE",
    "THIRD_PARTY.md": "THIRD_PARTY.md",
}

LAUNCHER = '@echo off\r\n"%~dp0anyxboxx-launcher.exe" --victory-exe "%~dp0victory.exe"\r\n'
TOPICS_FILE = ".github/beta-optimization-topics.txt"


class MergeConflict(Exception):
    def __init__(self, conflicts):
        super().__init__("Resolve these files on beta/optimizations before running the workflow again.")
        self.conflicts = conflicts


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def merge_commits(repo, base, incoming, source):
    if base == incoming:
        return base
    result = subprocess.run(
        ["git", "merge-tree", "--write-tree", "--name-only", "--no-messages",
         "-z", base, incoming], cwd=repo, capture_output=True, check=False,
    )
    tree, *paths = result.stdout.rstrip(b"\0").split(b"\0")
    if result.returncode == 1:
        raise MergeConflict({source: [path.decode("utf-8", errors="replace") for path in paths]})
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())

    return git(
        repo, "-c", "user.name=github-actions[bot]",
        "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit-tree", tree.decode("ascii"), "-p", base, "-p", incoming,
        "-m", f"chore: integrate {source} for Windows beta",
    )


def pending_topics(repo, optimizations):
    if not git(repo, "ls-tree", "--name-only", optimizations, "--", TOPICS_FILE):
        return
    entries = git(repo, "show", f"{optimizations}:{TOPICS_FILE}").splitlines()
    for entry in entries:
        entry = entry.partition("#")[0].strip()
        if not entry:
            continue
        fields = entry.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{40}", fields[0]):
            raise ValueError(f"Expected a full commit SHA and branch name in {TOPICS_FILE}")
        sha, branch = fields
        git(repo, "check-ref-format", f"refs/heads/{branch}")
        if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo).returncode:
            git(repo, "fetch", "--no-tags", "origin", sha)
        yield sha, branch


def integrate(repo, upstream, optimizations, branch):
    """Create a local integration ref after all pinned fork changes merge."""
    ref = f"refs/heads/{branch}"
    git(repo, "check-ref-format", ref)
    commit = merge_commits(repo, upstream, optimizations, "beta/optimizations")
    conflicts = {}
    for sha, source in pending_topics(repo, optimizations):
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, commit], cwd=repo, check=False,
        )
        if ancestry.returncode == 0:
            continue
        if ancestry.returncode != 1:
            ancestry.check_returncode()
        try:
            commit = merge_commits(repo, commit, sha, source)
        except MergeConflict as error:
            conflicts.update(error.conflicts)
    if conflicts:
        raise MergeConflict(conflicts)
    git(repo, "update-ref", ref, commit, "")
    return commit


def package(repo, output, revisions):
    """Archive the project binaries and the provenance of the tested checkout."""
    if git(repo, "rev-parse", "HEAD") != revisions["integration"]:
        raise ValueError("The package source does not match the tested integration commit.")
    for relative in BETA_FILES.values():
        if not (repo / relative).is_file():
            raise FileNotFoundError(f"Missing beta file {relative}")

    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for name, relative in BETA_FILES.items():
            archive.write(repo / relative, name)
        archive.writestr("Play-AnyXboxX.cmd", LAUNCHER)
        archive.writestr("revisions.json", json.dumps(revisions, indent=2) + "\n")

    with output.open("rb") as archive_file:
        digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    output.with_suffix(".zip.sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii", newline="\n",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    merge = commands.add_parser("integrate")
    merge.add_argument("upstream")
    merge.add_argument("optimizations")
    merge.add_argument("branch")

    archive = commands.add_parser("package")
    archive.add_argument("repo", type=Path)
    archive.add_argument("output", type=Path)
    archive.add_argument("upstream")
    archive.add_argument("optimizations")
    archive.add_argument("integration")
    args = parser.parse_args()

    if args.command == "package":
        package(args.repo, args.output, {
            "upstream": args.upstream,
            "optimizations": args.optimizations,
            "integration": args.integration,
        })
        return 0

    try:
        commit = integrate(Path.cwd(), args.upstream, args.optimizations, args.branch)
    except MergeConflict as error:
        lines = [str(error)]
        for source, paths in error.conflicts.items():
            lines.extend(("", source, *(f"  {path!r}" for path in paths)))
        report = "\n".join(lines)
        print(report, file=sys.stderr)
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary).open("a", encoding="utf-8") as stream:
                for line in report.splitlines():
                    print(f"    {line}" if line else "", file=stream)
        return 1
    print(commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
