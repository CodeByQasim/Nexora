"""
Phase 3 — Filesystem and Shell Sandbox Hardening tests
======================================================
Covers the Phase 3 hardening of the local sandbox adapter:

* filesystem access is confined to the approved workspace root
  (``..`` traversal, absolute/drive paths, symlink/junction escapes);
* credential/VCS material (``.ssh``, ``.env``, ``.git`` ...) is denied even
  inside the workspace;
* ``repo.list`` validates both the base directory and the glob pattern and
  hides sensitive/VCS entries;
* malformed paths and read/write/list resource limits fail closed;
* ``shell.run`` never executes raw model text through a shell: commands are
  tokenised shell-free, executables are resolved to trusted absolute paths,
  shells/inline interpreters and unapproved commands are refused, shell
  operators cannot chain a second command, and file arguments cannot escape
  the workspace;
* git tool arguments (branch names, ``git add`` file lists) cannot inject
  options or escape the workspace;
* the validator's test-runner path uses the same allowlisted, shell-free exec
  as the sandbox.

The existing baseline suites (test_token_optimizations,
test_model_router_and_adapter, test_phase1_security,
test_phase2_policy_and_tools) must keep passing alongside these tests.
"""
import asyncio
import json
import os
import subprocess

import pytest

from nexora.backend.app.core.tools import (
    LocalSandboxAdapter,
    ToolRouter,
    ToolValidationError,
    _build_approved_shell_argv,
    _is_safe_ref_name,
    _resolve_within_workspace,
    _validate_glob_pattern,
    PathBoundaryError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Filesystem boundary — positives (existing behaviour preserved)
# ---------------------------------------------------------------------------

class TestBoundaryHelpers:

    def test_resolve_within_workspace_accepts_child(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        resolved = _resolve_within_workspace(str(ws), "a/b.txt")
        assert resolved == (ws / "a" / "b.txt").resolve()

    @pytest.mark.parametrize(
        "bad", ["..", "../x", "/abs", "C:\\abs", "C:rel", "a\x00b", "", None]
    )
    def test_resolve_within_workspace_rejects(self, tmp_path, bad):
        ws = tmp_path / "ws"
        ws.mkdir()
        with pytest.raises(PathBoundaryError):
            _resolve_within_workspace(str(ws), bad)

    @pytest.mark.parametrize("bad", ["../*", "/abs/*", "a\x00b", "", "x/../../y"])
    def test_validate_glob_pattern_rejects(self, bad):
        with pytest.raises(PathBoundaryError):
            _validate_glob_pattern(bad)

    def test_validate_glob_pattern_accepts(self):
        assert _validate_glob_pattern("**/*.py") == "**/*.py"
        assert _validate_glob_pattern("sub/*.py") == "sub/*.py"


class TestFilesystemAuthorizedStillWorks:

    @pytest.mark.asyncio
    async def test_authorized_read_still_works(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write(ws / "src" / "app.py", "value = 1\n")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_read({"path": "src/app.py"}, str(ws))
        assert result["exit_code"] == 0
        assert "value = 1" in result["stdout"]

    @pytest.mark.asyncio
    async def test_authorized_write_and_create_still_works(self, tmp_path):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_write(
            {"path": "pkg/new.py", "content": "print('hi')\n", "mode": "create"},
            str(ws),
        )
        assert result["exit_code"] == 0
        assert (ws / "pkg" / "new.py").read_text(encoding="utf-8") == "print('hi')\n"

    @pytest.mark.asyncio
    async def test_authorized_list_still_works(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write(ws / "a.py", "x")
        _write(ws / "sub" / "b.py", "y")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_list({"path": ".", "pattern": "**/*.py"}, str(ws))
        assert result["exit_code"] == 0
        listed = json.loads(result["stdout"])
        assert "a.py" in listed
        assert "sub/b.py" in listed


# ---------------------------------------------------------------------------
# 2. Filesystem boundary — traversal / absolute / symlink escapes
# ---------------------------------------------------------------------------

class TestFilesystemBoundaryEnforced:

    @pytest.mark.asyncio
    async def test_parent_traversal_read_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write(tmp_path / "secret.txt", "TOP_SECRET")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_read({"path": "../secret.txt"}, str(ws))
        assert result["exit_code"] == 1
        assert result["stdout"] == ""
        assert "traversal" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_absolute_read_outside_workspace_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        outside = tmp_path / "secret.txt"
        _write(outside, "TOP_SECRET")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_read({"path": str(outside)}, str(ws))
        assert result["exit_code"] == 1
        assert "TOP_SECRET" not in result["stdout"]

    @pytest.mark.asyncio
    async def test_write_traversal_denied_without_side_effect(self, tmp_path):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_write(
            {"path": "../pwned.py", "content": "x"}, str(ws)
        )
        assert result["exit_code"] == 1
        assert not (tmp_path / "pwned.py").exists()

    @pytest.mark.asyncio
    async def test_symlink_file_escape_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        secret = tmp_path / "secret.txt"
        _write(secret, "TOP_SECRET")
        link = ws / "link.txt"
        try:
            os.symlink(secret, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this host")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_read({"path": "link.txt"}, str(ws))
        assert result["exit_code"] == 1
        assert "TOP_SECRET" not in result["stdout"]

    @pytest.mark.asyncio
    async def test_symlink_dir_escape_write_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        link = ws / "elsewhere"
        try:
            os.symlink(outside_dir, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this host")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_write(
            {"path": "elsewhere/pwned.py", "content": "x"}, str(ws)
        )
        assert result["exit_code"] == 1
        assert not (outside_dir / "pwned.py").exists()

    @pytest.mark.skipif(os.name != "nt", reason="Windows junction-specific check")
    @pytest.mark.asyncio
    async def test_junction_escape_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside / "secret.txt", "TOP_SECRET")
        link = ws / "link"
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
        )
        if created.returncode != 0 or not link.exists():
            pytest.skip("junction creation not permitted on this host")

        adapter = LocalSandboxAdapter()
        read = await adapter._repo_read({"path": "link/secret.txt"}, str(ws))
        assert read["exit_code"] == 1
        assert "TOP_SECRET" not in read["stdout"]

        write = await adapter._repo_write(
            {"path": "link/pwned.py", "content": "x"}, str(ws)
        )
        assert write["exit_code"] == 1
        assert not (outside / "pwned.py").exists()

    @pytest.mark.asyncio
    async def test_malformed_path_fails_closed(self, tmp_path):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        for bad in ("", None, "a\x00b"):
            result = await adapter._repo_read({"path": bad}, str(ws))
            assert result["exit_code"] == 1, bad


# ---------------------------------------------------------------------------
# 3. Filesystem boundary — sensitive material denied
# ---------------------------------------------------------------------------

class TestSensitivePathsDenied:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rel",
        [".env", ".ssh/id_rsa", ".git/config", ".aws/credentials", "id_ed25519"],
    )
    async def test_sensitive_read_denied(self, tmp_path, rel):
        ws = _make_workspace(tmp_path)
        _write(ws / rel, "SECRET_MATERIAL")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_read({"path": rel}, str(ws))
        assert result["exit_code"] == 1
        assert "SECRET_MATERIAL" not in result["stdout"]

    @pytest.mark.asyncio
    async def test_sensitive_write_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_write(
            {"path": ".ssh/authorized_keys", "content": "ssh-rsa AAAA"}, str(ws)
        )
        assert result["exit_code"] == 1
        assert not (ws / ".ssh" / "authorized_keys").exists()


# ---------------------------------------------------------------------------
# 4. repo.list boundary, sensitive hiding and limits
# ---------------------------------------------------------------------------

class TestRepoListBoundary:

    @pytest.mark.asyncio
    async def test_list_pattern_cannot_escape(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write(ws / "inside.txt", "i")
        _write(tmp_path / "outside.txt", "o")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_list(
            {"path": ".", "pattern": "../*.txt"}, str(ws)
        )
        assert result["exit_code"] == 1
        assert "outside.txt" not in result["stdout"]

    @pytest.mark.asyncio
    async def test_list_absolute_pattern_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_list(
            {"path": ".", "pattern": str(tmp_path / "*.txt")}, str(ws)
        )
        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_list_path_traversal_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_list({"path": ".."}, str(ws))
        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_list_sensitive_dir_denied(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write(ws / ".git" / "config", "x")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_list({"path": ".git"}, str(ws))
        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_list_hides_sensitive_and_vcs(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write(ws / "ok.py", "x")
        _write(ws / ".env", "SECRET=1")
        _write(ws / ".git" / "config", "x")
        adapter = LocalSandboxAdapter()
        result = await adapter._repo_list({"path": ".", "pattern": "**/*"}, str(ws))
        assert result["exit_code"] == 0
        listed = json.loads(result["stdout"])
        assert "ok.py" in listed
        assert not any(".env" in name or ".git" in name for name in listed)

    @pytest.mark.asyncio
    async def test_list_entry_limit_enforced(self, tmp_path, monkeypatch):
        import nexora.backend.app.core.tools as tools

        ws = _make_workspace(tmp_path)
        for index in range(5):
            _write(ws / f"f{index}.txt", "x")
        monkeypatch.setattr(tools, "MAX_LIST_ENTRIES", 2)
        adapter = tools.LocalSandboxAdapter()
        result = await adapter._repo_list({"path": ".", "pattern": "*.txt"}, str(ws))
        assert result["exit_code"] == 0
        assert len(json.loads(result["stdout"])) <= 2


# ---------------------------------------------------------------------------
# 5. Filesystem resource limits
# ---------------------------------------------------------------------------

class TestFilesystemLimits:

    @pytest.mark.asyncio
    async def test_read_size_limit_enforced(self, tmp_path, monkeypatch):
        import nexora.backend.app.core.tools as tools

        ws = _make_workspace(tmp_path)
        _write(ws / "big.txt", "x" * 100)
        monkeypatch.setattr(tools, "MAX_FILE_READ_BYTES", 10)
        adapter = tools.LocalSandboxAdapter()
        result = await adapter._repo_read({"path": "big.txt"}, str(ws))
        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_write_size_limit_enforced(self, tmp_path, monkeypatch):
        import nexora.backend.app.core.tools as tools

        ws = _make_workspace(tmp_path)
        monkeypatch.setattr(tools, "MAX_FILE_WRITE_BYTES", 10)
        adapter = tools.LocalSandboxAdapter()
        result = await adapter._repo_write(
            {"path": "big.txt", "content": "x" * 100}, str(ws)
        )
        assert result["exit_code"] == 1
        assert not (ws / "big.txt").exists()


# ---------------------------------------------------------------------------
# 6. Router-level pattern validation (defence in depth)
# ---------------------------------------------------------------------------

class TestRouterPatternValidation:

    def test_router_rejects_bad_list_pattern(self):
        router = ToolRouter()
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.list", {"pattern": "../*"}, "r1")

    def test_router_accepts_safe_list_pattern(self):
        router = ToolRouter()
        request = router.validate_and_route(
            "repo.list", {"path": ".", "pattern": "**/*.py"}, "r1", workspace_path="/ws"
        )
        assert request.tool_name.value == "repo.list"


# ---------------------------------------------------------------------------
# 7. shell.run — structured, shell-free execution
# ---------------------------------------------------------------------------

class TestShellRunStructured:

    @pytest.mark.asyncio
    async def test_reviewed_interpreter_command_runs(self, tmp_path):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._shell_run(
            {"command": "python -m pytest --version"}, str(ws), os.environ.copy()
        )
        assert result["exit_code"] == 0, result
        assert "Shell command rejected" not in result["stderr"]

    @pytest.mark.asyncio
    async def test_reviewed_git_command_runs(self, tmp_path):
        ws = _make_workspace(tmp_path)
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        adapter = LocalSandboxAdapter()
        result = await adapter._shell_run(
            {"command": "git status"}, str(ws), os.environ.copy()
        )
        assert result["exit_code"] == 0, result

    @pytest.mark.asyncio
    async def test_shell_run_uses_exec_not_shell(self, tmp_path, monkeypatch):
        captured = {}
        shell_calls = {"n": 0}

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (b"ok", b"")

        async def fake_exec(*argv, **kwargs):
            captured["argv"] = argv
            return _FakeProc()

        async def fake_shell(*args, **kwargs):
            shell_calls["n"] += 1
            raise AssertionError("create_subprocess_shell must never be used")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)

        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._shell_run(
            {"command": "python -m pytest -q"}, str(ws), os.environ.copy()
        )
        assert result["exit_code"] == 0
        assert shell_calls["n"] == 0
        assert os.path.isabs(captured["argv"][0])
        assert captured["argv"][1:] == ("-m", "pytest", "-q")

    @pytest.mark.asyncio
    async def test_relative_working_dir_is_bound_to_workspace(self, tmp_path, monkeypatch):
        ws = _make_workspace(tmp_path)
        (ws / "sub").mkdir()
        captured = {}

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (b"", b"")

        async def fake_exec(*argv, **kwargs):
            captured["cwd"] = kwargs.get("cwd")
            return _FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        adapter = LocalSandboxAdapter()
        result = await adapter._shell_run(
            {"command": "python -m pytest", "working_dir": "sub"},
            str(ws),
            os.environ.copy(),
        )
        assert result["exit_code"] == 0
        assert os.path.abspath(captured["cwd"]) == os.path.abspath(str(ws / "sub"))

    @pytest.mark.asyncio
    async def test_working_dir_escape_rejected(self, tmp_path):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._shell_run(
            {"command": "python -m pytest --version", "working_dir": "../"},
            str(ws),
            os.environ.copy(),
        )
        assert result["exit_code"] == 1
        assert "working directory" in result["stderr"].lower()


# ---------------------------------------------------------------------------
# 8. shell.run — shell operators never chain a second command
# ---------------------------------------------------------------------------

_SHELL_OPERATOR_PAYLOADS = [
    "python -m pytest --version; echo PWNED_MARKER",
    "python -m pytest --version && echo PWNED_MARKER",
    "python -m pytest --version || echo PWNED_MARKER",
    "ls | cat",
    "echo hi > pwned_marker.txt",
    "echo hi >> pwned_marker.txt",
    "cat < pwned_marker.txt",
    "echo $(whoami)",
    "echo `whoami`",
    "echo %USERNAME%",
    "echo hi ^& echo PWNED_MARKER",
    "python -m pytest --version\n echo PWNED_MARKER",
]


class TestShellOperatorRejection:

    @pytest.mark.parametrize("command", _SHELL_OPERATOR_PAYLOADS)
    def test_operator_command_rejected(self, command):
        with pytest.raises(ToolValidationError):
            _build_approved_shell_argv(command)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command", _SHELL_OPERATOR_PAYLOADS)
    async def test_operator_command_never_launches_a_subprocess(
        self, command, tmp_path, monkeypatch
    ):
        async def _boom(*args, **kwargs):
            raise AssertionError("no subprocess may be launched for a rejected command")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)
        monkeypatch.setattr(asyncio, "create_subprocess_shell", _boom)

        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._shell_run(
            {"command": command}, str(ws), os.environ.copy()
        )
        assert result["exit_code"] == 1, result
        assert "PWNED_MARKER" not in result["stdout"]
        assert not (ws / "pwned_marker.txt").exists()


# ---------------------------------------------------------------------------
# 9. shell.run — executable policy
# ---------------------------------------------------------------------------

class TestShellExecutablePolicy:

    @pytest.mark.parametrize(
        "command",
        [
            "bash -c 'echo hi'",
            "sh -c 'echo hi'",
            "cmd /c dir",
            "powershell -Command Get-ChildItem",
            "pwsh -c ls",
            "env python -m pytest",
            "xargs echo",
            "nohup ls",
        ],
    )
    def test_shells_and_launchers_rejected(self, command):
        with pytest.raises(ToolValidationError):
            _build_approved_shell_argv(command)

    @pytest.mark.parametrize(
        "command",
        [
            "python -c 'print(1)'",
            "python3 -m http.server",
            "py -c 'x'",
            "node -e 'console.log(1)'",
            "perl -e 'print 1'",
            "ruby -e 'puts 1'",
        ],
    )
    def test_inline_interpreters_rejected(self, command):
        with pytest.raises(ToolValidationError):
            _build_approved_shell_argv(command)

    @pytest.mark.parametrize(
        "command",
        [
            "curl http://evil.example.com",
            "wget http://evil.example.com",
            "rm -rf /",
            "sudo ls",
            "nc -l 1337",
            "git push origin main",
            "unknown-tool --flag",
        ],
    )
    def test_denied_or_unapproved_commands_rejected(self, command):
        with pytest.raises(ToolValidationError):
            _build_approved_shell_argv(command)

    @pytest.mark.parametrize(
        "command",
        [
            "cat /etc/passwd",
            "cat ../secret",
            "head C:\\Windows\\win.ini",
            "cat ..\\..\\secret",
        ],
    )
    def test_file_arguments_cannot_escape_workspace(self, command):
        with pytest.raises(ToolValidationError):
            _build_approved_shell_argv(command)

    def test_reviewed_command_produces_expected_argv(self):
        assert _build_approved_shell_argv("ls -la") == ["ls", "-la"]
        assert _build_approved_shell_argv("git status --short") == [
            "git", "status", "--short",
        ]

    def test_policy_unavailable_fails_closed(self, monkeypatch):
        import nexora.backend.app.core.tools as tools

        monkeypatch.setattr(tools, "_policy_command_sets", lambda: (set(), set()))
        with pytest.raises(ToolValidationError):
            tools._build_approved_shell_argv("ls")

    @pytest.mark.asyncio
    async def test_planted_executable_cannot_shadow(self, tmp_path, monkeypatch):
        ws = _make_workspace(tmp_path)
        (ws / "ls.bat").write_text("@echo off\r\necho PWNED_SHADOW\r\n", encoding="utf-8")
        monkeypatch.chdir(ws)
        monkeypatch.setenv("PATH", os.pathsep.join([".", os.curdir]))

        adapter = LocalSandboxAdapter()
        result = await adapter._shell_run(
            {"command": "ls"}, str(ws), os.environ.copy()
        )
        assert result["exit_code"] == 1
        assert "PWNED_SHADOW" not in result["stdout"]


# ---------------------------------------------------------------------------
# 10. shell.run — timeout / cancellation intact
# ---------------------------------------------------------------------------

class TestShellRunTimeout:

    @pytest.mark.asyncio
    async def test_timeout_raises_and_kills_process(self, tmp_path, monkeypatch):
        state = {"killed": False}

        class _SlowProc:
            returncode = 0

            async def communicate(self):
                await asyncio.sleep(5)
                return (b"", b"")

            def kill(self):
                state["killed"] = True

            async def wait(self):
                return 0

        async def fake_exec(*args, **kwargs):
            return _SlowProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        with pytest.raises(subprocess.TimeoutExpired):
            await adapter._shell_run(
                {"command": "python -m pytest", "timeout_seconds": 1},
                str(ws),
                os.environ.copy(),
            )
        assert state["killed"] is True


# ---------------------------------------------------------------------------
# 11. git argument hardening
# ---------------------------------------------------------------------------

class TestGitArgumentHardening:

    def test_ref_name_validator(self):
        assert _is_safe_ref_name("feature/add-thing") is True
        assert _is_safe_ref_name("main") is True
        for bad in ("-D", "--force", "a..b", "a b", "x~y", "x^y", "x:y", "a.lock", "", None):
            assert _is_safe_ref_name(bad) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["-D", "--force", "a..b", "branch name"])
    async def test_branch_option_injection_rejected(self, tmp_path, name):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._git_branch(
            {"action": "create", "name": name}, str(ws)
        )
        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("files", [["-A"], ["--"], ["../escape.py"], ["a.py", "-f"]])
    async def test_commit_file_injection_rejected(self, tmp_path, files):
        ws = _make_workspace(tmp_path)
        adapter = LocalSandboxAdapter()
        result = await adapter._git_commit(
            {"message": "msg", "files": files}, str(ws), os.environ.copy()
        )
        assert result["exit_code"] == 1


# ---------------------------------------------------------------------------
# 12. Validator test-runner path hardening
# ---------------------------------------------------------------------------

class TestValidatorRunnerHardening:

    def test_unapproved_test_command_rejected(self, tmp_path):
        from nexora.backend.app.core.validator import ValidatorCritic
        from nexora.backend.app.models import ValidationStatus

        validator = ValidatorCritic(str(tmp_path))
        result = validator._run_targeted_tests("curl http://evil.example.com")
        assert result.status == ValidationStatus.FAIL

    def test_injection_test_command_rejected(self, tmp_path):
        from nexora.backend.app.core.validator import ValidatorCritic
        from nexora.backend.app.models import ValidationStatus

        validator = ValidatorCritic(str(tmp_path))
        result = validator._run_targeted_tests("pytest -q; echo PWNED")
        assert result.status == ValidationStatus.FAIL

    def test_approved_test_command_still_runs(self, tmp_path):
        from nexora.backend.app.core.validator import ValidatorCritic
        from nexora.backend.app.models import ValidationStatus

        validator = ValidatorCritic(str(tmp_path))
        result = validator._run_targeted_tests("pytest --version")
        assert result.status == ValidationStatus.PASS
