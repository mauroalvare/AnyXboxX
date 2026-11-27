"""Prepare the Git integration ref and the Windows beta archive."""

import argparse
import hashlib
import json
import os
from pathlib import Path
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


class MergeConflict(Exception):
    def __init__(self, paths):
        super().__init__("Resolve these files on beta/optimizations before running the workflow again.")
        self.paths = paths


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def integrate(repo, upstream, optimizations, branch):
    """Create a new local branch with upstream and fork commits as its parents."""
    ref = f"refs/heads/{branch}"
    git(repo, "check-ref-format", ref)
    result = subprocess.run(
        ["git", "merge-tree", "--write-tree", "--name-only", "--no-messages",
         "-z", upstream, optimizations], cwd=repo, capture_output=True, check=False,
    )
    tree, *paths = result.stdout.rstrip(b"\0").split(b"\0")
    if result.returncode == 1:
        raise MergeConflict([path.decode("utf-8", errors="replace") for path in paths])
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())

    commit = upstream
    if upstream != optimizations:
        commit = git(
            repo, "-c", "user.name=github-actions[bot]",
            "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit-tree", tree.decode("ascii"), "-p", upstream, "-p", optimizations,
            "-m", "chore: integrate upstream and fork for Windows beta",
        )
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
        report = "\n".join((str(error), "", *(f"  {path!r}" for path in error.paths)))
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
