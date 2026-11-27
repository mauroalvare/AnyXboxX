import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from tools.beta_release import integrate, package


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "beta_release.py"


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def commit(repo, path, content):
    (repo / path).write_text(content, encoding="utf-8")
    git(repo, "add", "--", path)
    git(repo, "commit", "--quiet", "-m", f"Update {path}")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "--initial-branch=main", "--quiet")
    git(tmp_path, "config", "user.name", "Beta Test")
    git(tmp_path, "config", "user.email", "beta@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    commit(tmp_path, "shared file.txt", "base\n")
    git(tmp_path, "branch", "optimizations")
    return tmp_path


def test_integration_has_both_pinned_parents(repo):
    upstream = commit(repo, "upstream.txt", "upstream\n")
    git(repo, "switch", "--quiet", "optimizations")
    fork = commit(repo, "fork.txt", "optimization\n")
    branch = "beta/integration-321-1"

    sha = integrate(repo, upstream, fork, branch)

    assert git(repo, "rev-parse", branch) == sha
    assert git(repo, "show", "-s", "--format=%P", sha).split() == [upstream, fork]
    assert git(repo, "show", f"{sha}:upstream.txt") == "upstream"
    assert git(repo, "show", f"{sha}:fork.txt") == "optimization"
    assert git(repo, "rev-parse", "main", "optimizations").split() == [upstream, fork]


def test_conflict_reports_the_path_and_leaves_no_integration_branch(repo, monkeypatch):
    upstream = commit(repo, "shared file.txt", "upstream\n")
    git(repo, "switch", "--quiet", "optimizations")
    fork = commit(repo, "shared file.txt", "optimization\n")
    summary = repo / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "integrate", upstream, fork, "beta/integration-321-1"],
        cwd=repo, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 1
    assert "shared file.txt" in result.stderr
    assert "shared file.txt" in summary.read_text(encoding="utf-8")
    assert git(repo, "branch", "--list", "beta/*") == ""
    assert git(repo, "rev-parse", "HEAD") == fork


def test_package_contains_only_the_release_files_and_exact_revisions(repo):
    head = git(repo, "rev-parse", "HEAD")
    revisions = {"upstream": "a" * 40, "optimizations": "b" * 40, "integration": head}
    for name in (
        "target/release/anyxboxx.exe", "target/release/anyxboxx-launcher.exe",
        "third_party/victory/target/release/victory.exe", "LICENSE", "THIRD_PARTY.md",
        "target/release/local-test-output.dat",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    output = repo / "AnyXboxX-windows-x64.zip"

    package(repo, output, revisions)

    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "anyxboxx.exe", "anyxboxx-launcher.exe", "victory.exe", "LICENSE", "THIRD_PARTY.md",
            "Play-AnyXboxX.cmd", "revisions.json",
        }
        assert archive.read("victory.exe") == b"third_party/victory/target/release/victory.exe"
        assert b'--victory-exe "%~dp0victory.exe"' in archive.read("Play-AnyXboxX.cmd")
        assert json.loads(archive.read("revisions.json")) == revisions
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.with_suffix(".zip.sha256").read_bytes() == f"{digest}  {output.name}\n".encode()


def test_package_refuses_the_wrong_checkout(repo):
    output = repo / "AnyXboxX-windows-x64.zip"

    with pytest.raises(ValueError, match="does not match the tested integration commit"):
        package(repo, output, {"integration": "c" * 40})

    assert not output.exists()
