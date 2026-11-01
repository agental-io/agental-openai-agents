"""The client and session against a sandbox that is scripted: what they send the agental SDK, and what they make of its answers."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pytest
from agental import ApiError, AsyncAgental, AsyncSandbox
from agental.models import Command, ExecResult as AgentalExecResult, HostRoute
from agents.sandbox import ExecTimeoutError, ExecTransportError, ExposedPortUnavailableError, Manifest, WorkspaceReadNotFoundError
from agents.sandbox.errors import ConfigurationError, WorkspaceStartError

from agental_openai_agents import AgentalSandboxClient, AgentalSandboxClientOptions, AgentalSandboxSession, AgentalSandboxSessionState
from agental_openai_agents.sandbox import _newer, command_script, wire_command


def said(stdout: str = "", stderr: str = "", exit_code: int = 0, truncated: bool = False) -> AgentalExecResult:
    return AgentalExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr, truncated=truncated)


def command(status: str = "running", stdout: str = "", stderr: str = "", exit_code: int | None = None) -> Command:
    return Command(
        id="cmd-1", sandbox_id="sbx_1", pid=None, status=status, exit_code=exit_code, stdout=stdout, stderr=stderr, started_at="", finished_at=None
    )


@dataclass
class FakeFiles:
    written: dict[str, bytes] = field(default_factory=dict)
    held: dict[str, bytes] = field(default_factory=dict)

    async def read_bytes(self, path: str) -> bytes:
        if path not in self.held:
            raise ApiError(404, "file_not_found", f"{path} was not found.")
        return self.held[path]

    async def write(self, path: str, contents: str | bytes) -> None:
        self.written[path] = contents.encode() if isinstance(contents, str) else contents


@dataclass
class FakeCommands:
    readings: list[Command] = field(default_factory=list)
    started: list[list[str]] = field(default_factory=list)
    ran: list[tuple[list[str], float | None]] = field(default_factory=list)

    async def start(self, wire: list[str], *, timeout: float | None = None) -> Command:
        self.started.append(wire)
        return self.readings.pop(0)

    async def get(self, command_id: str) -> Command:
        return self.readings.pop(0)

    async def run(self, wire: list[str], *, timeout: float | None = None) -> Command:
        self.ran.append((wire, timeout))
        return self.readings.pop(0)


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sbx_1", status: str = "running") -> None:
        self.id = sandbox_id
        self.status = status
        self.files = FakeFiles()
        self.commands = FakeCommands()
        self.sent: list[tuple[list[str], float | None]] = []
        self.answer: Callable[[list[str]], AgentalExecResult] = lambda _: said()
        self.lifecycle: list[str] = []
        self.route_list: list[HostRoute] = []
        self.ttl: int | None = None
        self.started_with: dict[str, Any] = {}

    async def exec(self, wire: list[str], *, timeout: float | None = None) -> AgentalExecResult:
        self.sent.append((wire, timeout))
        return self.answer(wire)

    async def refresh(self) -> FakeSandbox:
        return self

    async def start(self, **options: Any) -> FakeSandbox:
        self.lifecycle.append("start")
        self.started_with = options
        self.status = "running"
        return self

    async def wait_until_running(self) -> FakeSandbox:
        self.lifecycle.append("wait")
        self.status = "running"
        return self

    async def stop(self) -> FakeSandbox:
        self.lifecycle.append("stop")
        self.status = "stopped"
        return self

    async def delete(self) -> None:
        self.lifecycle.append("delete")

    async def update(self, *, ttl_seconds: int | None = None) -> FakeSandbox:
        self.ttl = ttl_seconds
        return self

    async def routes(self) -> list[HostRoute]:
        return self.route_list

    async def expose(self, port: int, *, title: str | None = None, public: bool = True) -> HostRoute:
        route = HostRoute(
            id="rt_1", sandbox_id=self.id, port=port, title=title or "", public=public, hidden=False, proxy_path="/p",
            registered=True, listening=False, proxy="available", url=f"https://{port}-abc.agental.app",
        )
        self.route_list.append(route)
        return route


@dataclass
class FakeTemplate:
    id: str
    name: str


class FakeSandboxes:
    def __init__(self) -> None:
        self.made: list[dict[str, Any]] = []
        self.known: dict[str, FakeSandbox] = {}

    async def create(self, name: str | None = None, **options: Any) -> FakeSandbox:
        made = FakeSandbox(f"sbx_{len(self.made) + 1}")
        self.made.append({"name": name, **options})
        self.known[made.id] = made
        return made

    async def get(self, sandbox_id: str) -> FakeSandbox:
        if sandbox_id not in self.known:
            raise ApiError(404, "not_found", f"Sandbox {sandbox_id} was not found.")
        return self.known[sandbox_id]


class FakeTemplates:
    async def list(self) -> list[FakeTemplate]:
        return [FakeTemplate("tpl_a", "default"), FakeTemplate("tpl_b", "coding-agent")]


class FakeAgental:
    def __init__(self) -> None:
        self.sandboxes = FakeSandboxes()
        self.templates = FakeTemplates()


def client(fake: FakeAgental) -> AgentalSandboxClient:
    return AgentalSandboxClient(agental=cast(AsyncAgental, fake))


def session(fake: FakeSandbox | None = None, **state: Any) -> tuple[AgentalSandboxSession, FakeSandbox]:
    sandbox = fake or FakeSandbox()
    fields: dict[str, Any] = {"manifest": Manifest(root="/opt/agental/workspace"), "snapshot": {"type": "noop", "id": "s"}, "sandbox_id": sandbox.id}
    fields.update(state)
    inner = AgentalSandboxSession(state=AgentalSandboxSessionState(**fields), agental=cast(AsyncAgental, FakeAgental()), sandbox=cast(AsyncSandbox, sandbox))
    return inner, sandbox


def run_sh(wire: list[str]) -> str:
    """Runs what would be sent, here, to see that sh puts it back together."""
    return subprocess.run(wire, capture_output=True, text=True, check=True).stdout


# ----- the arguments a command is sent as -----


def test_a_command_that_fits_is_sent_as_it_is() -> None:
    assert wire_command(["echo", "a  b"], cwd=None, env={}) == ["echo", "a  b"]


def test_a_folder_and_variables_are_given_by_sh(tmp_path: Path) -> None:
    wire = wire_command(["sh", "-c", 'pwd; printf %s "$GREETING"'], cwd=str(tmp_path), env={"GREETING": "it's me"})
    assert wire is not None and wire[:2] == ["sh", "-c"]
    assert run_sh(wire) == f"{tmp_path.resolve()}\nit's me" or run_sh(wire) == f"{tmp_path}\nit's me"


def test_an_empty_argument_goes_through_sh_quoted() -> None:
    wire = wire_command(["printf", "[%s]", ""], cwd=None, env={})
    assert wire is not None and wire[:2] == ["sh", "-c"]
    assert run_sh(wire) == "[]"


def test_a_long_command_is_sent_in_pieces_that_sh_joins() -> None:
    text = "é" * 3000 + "😀" * 2000 + "x" * 50_000
    wire = wire_command(["printf", "%s", text], cwd=None, env={})
    assert wire is not None and wire[:3] == ["sh", "-c", 'IFS=; eval "$*"']
    assert all(len(each.encode("utf-16-le")) // 2 <= 4096 for each in wire)
    assert len(wire) <= 64
    assert run_sh(wire) == text


def test_one_too_long_for_arguments_is_left_to_a_file() -> None:
    assert wire_command(["printf", "%s", "x" * 300_000], cwd=None, env={}) is None


def test_a_process_id_is_written_down_when_asked(tmp_path: Path) -> None:
    pidfile = tmp_path / "ids" / "one.pid"
    wire = wire_command(["sh", "-c", "echo $$"], cwd=None, env={}, pidfile=str(pidfile))
    assert wire is not None
    assert run_sh(wire).strip() == pidfile.read_text().strip()


def test_the_script_execs_the_command() -> None:
    assert command_script(["ls", "-la"], cwd="/w", env={"A": "1"}) == "cd /w || exit 1\nexport A=1\nexec ls -la\n"


def test_new_output_is_found_after_the_api_drops_the_start_of_a_log() -> None:
    assert _newer("", "abc") == "abc"
    assert _newer("abc", "abcdef") == "def"
    seen = "x" * 600 + "marker-" + "y" * 500
    assert _newer(seen, seen[-800:] + "new") == "new"
    assert _newer("old text", "nothing alike").startswith("[agental: earlier output")


# ----- commands -----


def test_a_command_runs_in_the_workspace_with_the_manifests_variables() -> None:
    async def main() -> None:
        inner, sandbox = session(manifest=Manifest(root="/opt/agental/workspace", environment={"value": {"MODE": "test"}}), workspace_root_ready=True)
        sandbox.answer = lambda _: said(stdout="ok\n")
        result = await inner.exec("echo ok", timeout=5)
        assert (result.stdout, result.exit_code) == (b"ok\n", 0)
        wire, timeout = sandbox.sent[0]
        assert wire[:2] == ["sh", "-c"] and timeout == 5
        assert wire[2] == "cd /opt/agental/workspace || exit 1\nexport MODE=test\nexec sh -lc 'echo ok'\n"

    asyncio.run(main())


def test_a_command_that_runs_out_of_time_is_a_timeout() -> None:
    async def main() -> None:
        inner, sandbox = session()

        def slow(_: list[str]) -> AgentalExecResult:
            import time

            time.sleep(0.25)
            return said(exit_code=124)

        sandbox.answer = slow
        with pytest.raises(ExecTimeoutError):
            await inner.exec("sleep 9", timeout=0.2)
        # A command that exits 124 by itself, quickly, is not one that ran out.
        sandbox.answer = lambda _: said(exit_code=124)
        assert (await inner.exec("exit 124", timeout=30)).exit_code == 124

    asyncio.run(main())


def test_no_timeout_is_the_apis_ten_minutes_run_in_the_background() -> None:
    async def main() -> None:
        inner, sandbox = session()
        sandbox.commands.readings = [command("succeeded", stdout="built\n", exit_code=0)]
        result = await inner.exec("make", timeout=None)
        assert (result.stdout, result.exit_code) == (b"built\n", 0)
        assert sandbox.sent == [] and sandbox.commands.ran[0][1] == 600.0
        sandbox.commands.readings = [command("timed_out", exit_code=124)]
        with pytest.raises(ExecTimeoutError):
            await inner.exec("make", timeout=1200)

    asyncio.run(main())


def test_output_the_api_cut_says_so() -> None:
    async def main() -> None:
        inner, sandbox = session()
        sandbox.answer = lambda _: said(stdout="a" * 10, stderr="e", truncated=True)
        result = await inner.exec("yes", timeout=5)
        assert result.stderr.startswith(b"e\n[agental: the output of this command was cut at 1 MB]")

    asyncio.run(main())


def test_a_refused_command_is_a_transport_error() -> None:
    async def main() -> None:
        inner, sandbox = session()

        def refuse(_: list[str]) -> AgentalExecResult:
            raise ApiError(409, "sandbox_not_running", "Sandbox is stopped.")

        sandbox.answer = refuse
        with pytest.raises(ExecTransportError) as raised:
            await inner.exec("true", timeout=5)
        assert raised.value.context["code"] == "sandbox_not_running"

    asyncio.run(main())


def test_a_long_line_of_shell_is_run_from_a_file() -> None:
    async def main() -> None:
        inner, sandbox = session()
        await inner.exec("printf %s " + "x" * 150_000, timeout=5)
        (path, script), = sandbox.files.written.items()
        assert path.startswith(".openai-agents/run-") and script == b'rm -f -- "$0"\nprintf %s ' + b"x" * 150_000
        # sh -lc stays a login shell: sh -l on the file.
        assert sandbox.sent[0][0] == ["sh", "-l", f"/opt/agental/{path}"]

    asyncio.run(main())


def test_a_command_too_long_for_arguments_is_written_to_a_file_first() -> None:
    async def main() -> None:
        inner, sandbox = session()
        await inner.exec("printf", "%s", "x" * 300_000, shell=False, timeout=5)
        (path, script), = sandbox.files.written.items()
        assert path.startswith(".openai-agents/run-") and script.startswith(b'rm -f -- "$0"\nexec printf %s xxx')
        assert sandbox.sent[0][0] == ["sh", f"/opt/agental/{path}"]

    asyncio.run(main())


def test_a_command_left_running_is_read_until_it_ends() -> None:
    async def main() -> None:
        inner, sandbox = session()
        sandbox.commands.readings = [command(stdout="one\n")] + [command(stdout="one\ntwo\n")] * 8 + [command("succeeded", stdout="one\ntwo\nthree\n", exit_code=0)]
        first = await inner.pty_exec_start("npm test", yield_time_s=0.25)
        assert first.process_id is not None and first.exit_code is None
        assert "echo $$ > /tmp/agental-openai-agents/" in sandbox.commands.started[0][2]
        with pytest.raises(ExecTransportError, match="cannot type into"):
            await inner.pty_write_stdin(session_id=first.process_id, chars="y\n")
        last = await inner.pty_write_stdin(session_id=first.process_id, chars="")
        assert (last.process_id, last.exit_code) == (None, 0)
        assert first.output + last.output == b"one\ntwo\nthree\n"

    asyncio.run(main())


# ----- files -----


def test_a_file_in_the_files_folder_is_read_by_the_files_api() -> None:
    async def main() -> None:
        inner, sandbox = session()
        sandbox.files.held["workspace/a.txt"] = b"hello"
        assert await inner._read_file(PurePosixPath("/opt/agental/workspace/a.txt"), shown=Path("a.txt")) == b"hello"
        with pytest.raises(WorkspaceReadNotFoundError):
            await inner._read_file(PurePosixPath("/opt/agental/workspace/none.txt"), shown=Path("none.txt"))
        assert sandbox.sent == []

    asyncio.run(main())


def test_a_file_elsewhere_is_read_in_parts_through_a_copy() -> None:
    async def main() -> None:
        inner, sandbox = session()
        sandbox.answer = lambda wire: said(stdout="part-aaaa\npart-aaab\n") if "split" in " ".join(wire) else said()

        async def read_part(path: str) -> bytes:
            return b"A" if path.endswith("aaaa") else b"B"

        sandbox.files.read_bytes = read_part  # type: ignore[method-assign]
        assert await inner._read_file(PurePosixPath("/etc/hosts"), shown=Path("/etc/hosts")) == b"AB"
        assert sandbox.sent[-1][0][:3] == ["rm", "-rf", "--"]
        sandbox.answer = lambda wire: said(exit_code=3)
        with pytest.raises(WorkspaceReadNotFoundError):
            await inner._read_file(PurePosixPath("/etc/none"), shown=Path("/etc/none"))

    asyncio.run(main())


def test_a_write_goes_through_a_copy_so_the_file_keeps_its_mode() -> None:
    async def main() -> None:
        inner, sandbox = session()
        await inner._write_file(PurePosixPath("/opt/agental/workspace/run.sh"), b"x" * 9_000_000, shown=Path("run.sh"))
        parts = sorted(sandbox.files.written)
        assert [len(sandbox.files.written[each]) for each in parts] == [8_000_000, 1_000_000]
        wire = sandbox.sent[0][0]
        assert 'cat "$2"/part-* > "$1"' in wire[2] and wire[4] == "/opt/agental/workspace/run.sh"

    asyncio.run(main())


# ----- the sandbox's life -----


def test_a_new_sandbox_is_made_from_the_options_and_deleted_at_the_end() -> None:
    async def main() -> None:
        fake = FakeAgental()
        made = await client(fake).create(options=AgentalSandboxClientOptions(template="coding-agent", size="medium", allow=("192.0.2.9",), ttl_seconds=7200))
        inner = cast(AgentalSandboxSession, made._inner)
        assert fake.sandboxes.made[0]["template"] == "tpl_b" and fake.sandboxes.made[0]["size"] == "medium"
        assert fake.sandboxes.made[0]["allow"] == ["192.0.2.9"] and fake.sandboxes.made[0]["name"].startswith("openai-agents-")
        assert inner.state.manifest.root == "/opt/agental/workspace"
        sandbox = cast(FakeSandbox, inner.sandbox)
        # The time limit is sent with the create, so a failure after it cannot leave a sandbox with none.
        assert fake.sandboxes.made[0]["ttl_seconds"] == 7200 and sandbox.ttl is None
        await inner.shutdown()
        await client(fake).delete(made)
        assert sandbox.lifecycle == ["delete"]

    asyncio.run(main())


def test_no_time_limit_given_leaves_it_to_the_template() -> None:
    async def main() -> None:
        fake = FakeAgental()
        await client(fake).create(options=AgentalSandboxClientOptions())
        assert "ttl_seconds" not in fake.sandboxes.made[0]

    asyncio.run(main())


def test_region_keys_and_burst_limits_are_sent_only_when_given() -> None:
    async def main() -> None:
        fake = FakeAgental()
        await client(fake).create(options=AgentalSandboxClientOptions())
        for left_out in ("region", "ssh_keys", "max_burst_fee_micros", "max_burst_wait"):
            assert left_out not in fake.sandboxes.made[0]
        options = AgentalSandboxClientOptions(region=("eu", "us"), ssh_keys=("ssh-ed25519 AAAA me",), max_burst_fee_micros=0, max_burst_wait=30.0)
        await client(fake).create(options=options)
        made = fake.sandboxes.made[1]
        assert made["region"] == ("eu", "us") and made["ssh_keys"] == ["ssh-ed25519 AAAA me"]
        assert made["max_burst_fee_micros"] == 0 and made["max_burst_wait"] == 30.0
        await client(fake).create(options=AgentalSandboxClientOptions(region="eu"))
        assert fake.sandboxes.made[2]["region"] == "eu"
        with pytest.raises(ConfigurationError, match="region, ssh_keys are for a new sandbox"):
            await client(fake).create(options=AgentalSandboxClientOptions(sandbox_id="sbx_1", region="eu", ssh_keys=("k",)))

    asyncio.run(main())


def test_a_client_passed_in_is_not_closed() -> None:
    class Closable(FakeAgental):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    async def main() -> None:
        fake = Closable()
        await client(fake).aclose()
        assert not fake.closed

    asyncio.run(main())


def test_a_sandbox_given_by_its_id_is_left_running_or_stopped() -> None:
    async def main() -> None:
        fake = FakeAgental()
        there = FakeSandbox("sbx_there")
        fake.sandboxes.known[there.id] = there
        left = await client(fake).create(options=AgentalSandboxClientOptions(sandbox_id="sbx_there"))
        await cast(AgentalSandboxSession, left._inner).shutdown()
        assert there.lifecycle == []
        stopped = await client(fake).create(options=AgentalSandboxClientOptions(sandbox_id="sbx_there", stop_on_exit=True))
        await cast(AgentalSandboxSession, stopped._inner).shutdown()
        assert there.lifecycle == ["stop"]
        with pytest.raises(ConfigurationError, match="size is for a new sandbox"):
            await client(fake).create(options=AgentalSandboxClientOptions(sandbox_id="sbx_there", size="large"))
        with pytest.raises(WorkspaceStartError):
            await client(fake).create(options=AgentalSandboxClientOptions(template="nonesuch"))

    asyncio.run(main())


def test_a_stopped_sandbox_is_started_when_the_session_starts() -> None:
    async def main() -> None:
        inner, sandbox = session(FakeSandbox(status="stopped"), max_burst_fee_micros=500_000)
        await inner._ensure_backend_started()
        assert sandbox.lifecycle == ["start"]
        assert sandbox.started_with == {"max_burst_fee_micros": 500_000, "max_burst_wait": None}

    asyncio.run(main())


def test_resume_finds_the_sandbox_or_makes_a_new_one() -> None:
    async def main() -> None:
        fake = FakeAgental()
        agental_client = client(fake)
        options = AgentalSandboxClientOptions(stop_on_exit=True, region="eu", ssh_keys=("ssh-ed25519 AAAA me",), max_burst_fee_micros=0)
        made = await agental_client.create(options=options)
        payload = json.loads(json.dumps(agental_client.serialize_session_state(made.state)))
        state = agental_client.deserialize_session_state(payload)
        assert isinstance(state, AgentalSandboxSessionState) and state.sandbox_id == "sbx_1" and state.stop_on_exit
        assert (state.region, state.ssh_keys, state.max_burst_fee_micros) == ("eu", ("ssh-ed25519 AAAA me",), 0)
        again = await agental_client.resume(state)
        assert cast(AgentalSandboxSession, again._inner).sandbox is fake.sandboxes.known["sbx_1"]

        del fake.sandboxes.known["sbx_1"]
        replaced = await agental_client.resume(agental_client.deserialize_session_state(payload))
        assert replaced.state.sandbox_id == "sbx_2" and replaced.state.workspace_root_ready is False
        # The new one is made as the first was.
        assert fake.sandboxes.made[1]["region"] == "eu" and fake.sandboxes.made[1]["ssh_keys"] == ["ssh-ed25519 AAAA me"]
        assert fake.sandboxes.made[1]["max_burst_fee_micros"] == 0

        given = AgentalSandboxSessionState(manifest=Manifest(), snapshot={"type": "noop", "id": "s"}, sandbox_id="sbx_gone", created_by_us=False)
        with pytest.raises(WorkspaceStartError):
            await agental_client.resume(given)

    asyncio.run(main())


def test_a_port_is_given_its_public_address() -> None:
    async def main() -> None:
        inner, sandbox = session(exposed_ports=(8080,))
        endpoint = await inner.resolve_exposed_port(8080)
        assert (endpoint.host, endpoint.port, endpoint.tls) == ("8080-abc.agental.app", 443, True)
        assert len(sandbox.route_list) == 1
        await inner.resolve_exposed_port(8080)
        assert len(sandbox.route_list) == 1
        sandbox.route_list[0].url = None
        with pytest.raises(ExposedPortUnavailableError):
            await inner.resolve_exposed_port(8080)

    asyncio.run(main())


def test_the_client_says_when_there_is_no_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AGENTAL_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with pytest.raises(ConfigurationError, match="No API key"):
        AgentalSandboxClient()


def test_options_go_through_json_and_back() -> None:
    for region in ("eu", ("eu", "us")):
        options = AgentalSandboxClientOptions(size="tiny", region=region, ssh_keys=("ssh-ed25519 AAAA me",), exposed_ports=(3000,), stop_on_exit=True)
        parsed = AgentalSandboxClientOptions.parse(json.loads(json.dumps(options.model_dump(mode="json"))))
        assert parsed == options and isinstance(io.BytesIO(), io.IOBase)
