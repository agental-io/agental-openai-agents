"""agental sandboxes for the OpenAI Agents SDK.

A SandboxAgent's commands run in an agental sandbox through the API's exec and its
background commands, and its files are read and written through the API's files API,
which reaches /opt/agental on the sandbox. The workspace is /opt/agental/workspace unless
the manifest names another root. A file elsewhere goes through a copy under
/opt/agental/.openai-agents, which is made and taken away again by the same calls.

What the API does not do, this module does around it:

- A command is given its working folder and the manifest's variables by a short sh script,
  as the API takes neither.
- The API takes at most 64 arguments of 4096 characters. A longer command is sent in
  pieces that sh puts back together, and one too long for that is written to a file first.
  So is a line of shell over 100 KB, as Linux runs no program with an argument over 128 KB.
- A command runs for at most ten minutes. One allowed more than 90 seconds is run in the
  background and read until it ends, as a request held longer is cut by the proxy in front.
- A file is read and written up to 10 MB at once. A larger one is sent in parts.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import shlex
import tarfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from agental import AgentalError, ApiError, AsyncAgental, AsyncSandbox
from agental.models import Command
from agents.sandbox import (
    ErrorCode,
    ExecResult,
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortEndpoint,
    ExposedPortUnavailableError,
    Manifest,
    SnapshotSpec,
    User,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceWriteTypeError,
    resolve_snapshot,
)
from agents.sandbox.errors import ConfigurationError, WorkspaceStartError
from agents.sandbox.session import (
    BaseSandboxClient,
    BaseSandboxClientOptions,
    BaseSandboxSession,
    Dependencies,
    Instrumentation,
    SandboxSession,
    SandboxSessionState,
)
from agents.sandbox.session.pty_types import (
    PTY_PROCESSES_MAX,
    PtyExecUpdate,
    allocate_pty_process_id,
    clamp_pty_yield_time_ms,
    process_id_to_prune_from_meta,
    resolve_pty_write_yield_time_ms,
    truncate_text_by_tokens,
)
from agents.sandbox.session.runtime_helpers import RESOLVE_WORKSPACE_PATH_HELPER, RuntimeHelperScript
from agents.sandbox.snapshot import SnapshotBase
from agents.sandbox.util import UnsafeTarMemberError, validate_tarfile
from agents.sandbox.workspace_paths import coerce_posix_path, posix_path_as_path, sandbox_path_str

logger = logging.getLogger(__name__)

Size = Literal["tiny", "small", "medium", "large"]

#: Where the files API reads and writes on every agental sandbox.
FILES_ROOT = "/opt/agental"
#: The folder under the files root that copies of files and archives pass through.
STAGING = ".openai-agents"

_DEFAULT_MANIFEST_ROOT = cast(str, Manifest.model_fields["root"].default)
#: The API's longest argument, counted in UTF-16 code units as its JSON is.
_ARG_LIMIT = 4096
_ARG_COUNT = 64
#: A line of shell longer than this, in bytes, is run from a file: Linux takes no argument over 128 KB.
_SHELL_TEXT_LIMIT = 100_000
#: No command the API runs lives longer.
_COMMAND_LIMIT_S = 600.0
#: A request held open longer than this may be cut by the proxy in front of the API.
_HELD_LIMIT_S = 90.0
#: The size of each part a large file is sent or read in, under the API's 10 MB.
_PART_BYTES = 8_000_000
#: A background command's output is kept as its last 64 KB on each stream.
_LOG_LIMIT = 65_536
_POLL_S = 0.25
_NOT_FOUND = 3
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CUT = b"\n[agental: the output of this command was cut at 1 MB]\n"
_DROPPED = "[agental: earlier output of this command was dropped]\n"


class AgentalSandboxClientOptions(BaseSandboxClientOptions):
    """What sandbox a session runs in.

    Left as they are, a new sandbox is made from the account's default template at that
    template's size, and deleted when the session ends.
    """

    type: Literal["agental"] = "agental"
    #: An existing sandbox to run in, in place of a new one. It is not deleted when the session ends.
    sandbox_id: str | None = None
    #: The new sandbox's name: lowercase letters, digits and hyphens. Made up when left out.
    name: str | None = None
    #: The template, by its id or its name, that the new sandbox is made from and held to.
    template: str | None = None
    #: tiny, small, medium or large. Left out, the template's.
    size: Size | None = None
    #: One of the account's snapshots to make the new sandbox from.
    snapshot_id: str | None = None
    #: Addresses the new sandbox may reach on top of its template's outbound rule.
    allow: tuple[str, ...] | None = None
    #: Seconds the new sandbox runs after a start before the API stops it. Left out, its template's start says, and
    #: with none there the deployment's default, which on agental.io is none: it runs until it is stopped.
    ttl_seconds: int | None = None
    #: The region to make the new sandbox in, as "eu", or several to try in order, as ("eu", "us"). Left out, the
    #: default region.
    region: str | tuple[str, ...] | None = None
    #: OpenSSH public key lines let in as root on the new sandbox, on top of the keys its template lets in.
    ssh_keys: tuple[str, ...] | None = None
    #: The most a create or a start may pay in burst fees, in millionths of a US dollar: 0 refuses any fee. Left out,
    #: what the account's monthly burst budget allows.
    max_burst_fee_micros: int | None = None
    #: Seconds a create or a start refused for its burst fee may wait, when the API names a time it would cost nothing
    #: by, to be tried once more then. Left out, it is not tried again.
    max_burst_wait: float | None = None
    #: Ports that get a public address when the agent asks for them.
    exposed_ports: tuple[int, ...] = ()
    #: Stop the sandbox when the session ends, in place of deleting a new one or leaving an existing one running.
    stop_on_exit: bool = False


class AgentalSandboxSessionState(SandboxSessionState):
    """What a session is resumed from. The API key is not in it: resume() is given it by its client."""

    type: Literal["agental"] = "agental"
    sandbox_id: str
    created_by_us: bool = True
    files_root: str = FILES_ROOT
    name: str | None = None
    template: str | None = None
    size: Size | None = None
    snapshot_id: str | None = None
    allow: tuple[str, ...] | None = None
    ttl_seconds: int | None = None
    region: str | tuple[str, ...] | None = None
    ssh_keys: tuple[str, ...] | None = None
    max_burst_fee_micros: int | None = None
    max_burst_wait: float | None = None
    stop_on_exit: bool = False


@dataclass
class _Running:
    """A command the agent left running, read through the API's background commands."""

    command_id: str
    pidfile: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    last_used: float = field(default_factory=time.monotonic)


class AgentalSandboxSession(BaseSandboxSession):
    """A session in one agental sandbox."""

    state: AgentalSandboxSessionState

    def __init__(self, *, state: AgentalSandboxSessionState, agental: AsyncAgental, sandbox: AsyncSandbox) -> None:
        self.state = state
        self._agental = agental
        self._sandbox = sandbox
        self._root_made = False
        self._ended = False
        self._pty_lock = asyncio.Lock()
        self._pty_processes: dict[int, _Running] = {}
        self._reserved_pty_process_ids: set[int] = set()

    @property
    def sandbox(self) -> AsyncSandbox:
        """The sandbox, to use with the agental SDK."""
        return self._sandbox

    # ----- start and end -----

    def _runtime_helpers(self) -> tuple[RuntimeHelperScript, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    def _current_runtime_helper_cache_key(self) -> object | None:
        return self.state.sandbox_id

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        return await self._validate_remote_path_access(path, for_write=for_write)

    async def _ensure_backend_started(self) -> None:
        root = posix_path_as_path(coerce_posix_path(self.state.manifest.root))
        try:
            await self._sandbox.refresh()
            status = self._sandbox.status
            if status == "stopped":
                await self._sandbox.start(max_burst_fee_micros=self.state.max_burst_fee_micros, max_burst_wait=self.state.max_burst_wait)
            elif status in ("error", "deleted"):
                raise WorkspaceStartError(path=root, context=self._context(reason=f"sandbox_{status}"))
            elif status != "running":
                await self._sandbox.wait_until_running()
        except AgentalError as error:
            raise WorkspaceStartError(path=root, context=self._context(error), cause=error) from error

    async def _prepare_backend_workspace(self) -> None:
        root = self._workspace_root_path()
        result = await self._run(["mkdir", "-p", "--", sandbox_path_str(root)], cwd=None, env={}, timeout=60.0)
        if not result.ok():
            raise WorkspaceStartError(path=root, context=self._context(**_said(result)))
        self._root_made = True

    async def running(self) -> bool:
        try:
            await self._sandbox.refresh()
        except AgentalError:
            return False
        return self._sandbox.status == "running"

    async def _shutdown_backend(self) -> None:
        # The runner ends a session and then deletes it through its client, which ends it again.
        if self._ended:
            return
        self._ended = True
        try:
            if self.state.stop_on_exit:
                await self._sandbox.stop()
            elif self.state.created_by_us:
                await self._sandbox.delete()
        except ApiError as error:
            if error.status not in (404, 409):
                logger.warning("Could not end agental sandbox %s: %s", self.state.sandbox_id, error)
        except AgentalError as error:
            logger.warning("Could not end agental sandbox %s: %s", self.state.sandbox_id, error)

    # ----- commands -----

    async def _exec_internal(self, *command: str | Path, timeout: float | None = None) -> ExecResult:
        return await self._run([str(part) for part in command], cwd=self._cwd(), env=await self._env(), timeout=timeout)

    def _cwd(self) -> str | None:
        if self._root_made or self.state.workspace_root_ready:
            return sandbox_path_str(self._workspace_root_path())
        return None

    async def _env(self) -> dict[str, str]:
        variables = await self.state.manifest.environment.resolve()
        for name in variables:
            if not _ENV_NAME.match(name):
                raise ConfigurationError(
                    message=f"agental cannot give a command the variable {name!r}: a name is letters, digits and underscores",
                    error_code=ErrorCode.SANDBOX_CONFIG_INVALID,
                    op="exec",
                    context={"backend": "agental", "variable": name},
                )
        return dict(variables)

    async def _run(self, argv: list[str], *, cwd: str | None, env: Mapping[str, str], timeout: float | None) -> ExecResult:
        limit = _COMMAND_LIMIT_S if timeout is None else min(max(float(timeout), 0.1), _COMMAND_LIMIT_S)
        wire = await self._wire_for(argv, cwd=cwd, env=env)
        started = time.monotonic()
        try:
            if limit <= _HELD_LIMIT_S:
                said = await self._sandbox.exec(wire, timeout=limit)
                timed_out = said.exit_code == 124 and time.monotonic() - started >= limit * 0.9
                stdout, stderr, exit_code = said.stdout, said.stderr, said.exit_code
                cut = said.truncated
            else:
                ran = await self._sandbox.commands.run(wire, timeout=limit)
                timed_out = ran.status == "timed_out"
                stdout, stderr = ran.stdout, ran.stderr
                exit_code = ran.exit_code if ran.exit_code is not None else 1
                cut = len(stdout) >= _LOG_LIMIT or len(stderr) >= _LOG_LIMIT
        except AgentalError as error:
            raise ExecTransportError(command=argv, context=self._context(error), cause=error, retryable=_retryable(error)) from error
        if timed_out:
            raise ExecTimeoutError(command=argv, timeout_s=timeout if timeout is not None else limit, context=self._context(limit_s=limit))
        stderr_bytes = stderr.encode("utf-8", errors="replace")
        if cut:
            stderr_bytes += _CUT
        return ExecResult(stdout=stdout.encode("utf-8", errors="replace"), stderr=stderr_bytes, exit_code=exit_code)

    async def _wire_for(self, argv: Sequence[str], *, cwd: str | None, env: Mapping[str, str], pidfile: str | None = None) -> list[str]:
        """The arguments the API is sent for a command: the command itself when it can be, or sh around it."""
        if len(argv) == 3 and argv[1] in ("-c", "-lc") and len(argv[2].encode("utf-8", errors="surrogatepass")) > _SHELL_TEXT_LIMIT:
            # Linux runs no program with one argument over 128 KB, so a long line of shell is run from a file instead.
            name = f"run-{uuid.uuid4().hex}.sh"
            try:
                await self._sandbox.files.write(f"{STAGING}/{name}", 'rm -f -- "$0"\n' + argv[2])
            except AgentalError as error:
                raise ExecTransportError(command=list(argv), context=self._context(error), cause=error) from error
            argv = [argv[0], *(["-l"] if argv[1] == "-lc" else []), f"{self._staging()}/{name}"]
        wire = wire_command(argv, cwd=cwd, env=env, pidfile=pidfile)
        if wire is not None:
            return wire
        # Too long to go as arguments: written to a file that sh runs and takes away first.
        name = f"run-{uuid.uuid4().hex}.sh"
        script = 'rm -f -- "$0"\n' + command_script(argv, cwd=cwd, env=env, pidfile=pidfile)
        try:
            await self._sandbox.files.write(f"{STAGING}/{name}", script)
        except AgentalError as error:
            raise ExecTransportError(command=list(argv), context=self._context(error), cause=error) from error
        return ["sh", f"{self._staging()}/{name}"]

    def supports_pty(self) -> bool:
        return True

    async def pty_exec_start(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
        tty: bool = False,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        # A command is left running in the background and its output read as it comes. There is no
        # terminal: tty is asked for and not given, and nothing can be typed into a command once it runs.
        argv = self._prepare_exec_command(*command, shell=shell, user=user)
        pidfile = f"/tmp/agental-openai-agents/{uuid.uuid4().hex}.pid"
        wire = await self._wire_for(argv, cwd=self._cwd(), env=await self._env(), pidfile=pidfile)
        limit = _COMMAND_LIMIT_S if timeout is None else min(max(float(timeout), 0.1), _COMMAND_LIMIT_S)
        try:
            started = await self._sandbox.commands.start(wire, timeout=limit)
        except AgentalError as error:
            raise ExecTransportError(command=argv, context=self._context(error), cause=error, retryable=_retryable(error)) from error
        entry = _Running(command_id=started.id, pidfile=pidfile)
        async with self._pty_lock:
            process_id = allocate_pty_process_id(self._reserved_pty_process_ids)
            self._reserved_pty_process_ids.add(process_id)
            pruned = self._prune_pty_processes()
            self._pty_processes[process_id] = entry
        if pruned is not None:
            await self._terminate([pruned])
        yield_ms = clamp_pty_yield_time_ms(10_000 if yield_time_s is None else int(yield_time_s * 1000))
        output, count = await self._collect(entry, first=started, yield_ms=yield_ms, max_output_tokens=max_output_tokens)
        return await self._update(process_id, entry, output, count)

    async def pty_write_stdin(
        self,
        *,
        session_id: int,
        chars: str,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        async with self._pty_lock:
            entry = self._resolve_pty_session_entry(pty_processes=self._pty_processes, session_id=session_id)
        if chars:
            raise ExecTransportError(
                command=("<write_stdin>",),
                message="agental cannot type into a command that is running: give it its input in the command, with a pipe or a here-document",
                context=self._context(session_id=session_id),
                retryable=False,
            )
        yield_ms = resolve_pty_write_yield_time_ms(yield_time_ms=250 if yield_time_s is None else int(yield_time_s * 1000), input_empty=True)
        output, count = await self._collect(entry, first=None, yield_ms=yield_ms, max_output_tokens=max_output_tokens)
        entry.last_used = time.monotonic()
        return await self._update(session_id, entry, output, count)

    async def pty_terminate_all(self) -> None:
        async with self._pty_lock:
            entries = list(self._pty_processes.values())
            self._pty_processes.clear()
            self._reserved_pty_process_ids.clear()
        await self._terminate(entries)

    async def _collect(self, entry: _Running, *, first: Command | None, yield_ms: int, max_output_tokens: int | None) -> tuple[bytes, int | None]:
        deadline = time.monotonic() + yield_ms / 1000
        output: list[str] = []
        command = first
        while True:
            if command is None:
                try:
                    command = await self._sandbox.commands.get(entry.command_id)
                except AgentalError as error:
                    raise ExecTransportError(command=("<read_command>",), context=self._context(error), cause=error, retryable=_retryable(error)) from error
            output.append(_newer(entry.stdout, command.stdout))
            output.append(_newer(entry.stderr, command.stderr))
            entry.stdout, entry.stderr = command.stdout, command.stderr
            if command.status != "running":
                entry.exit_code = command.exit_code if command.exit_code is not None else 124 if command.status == "timed_out" else 1
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(_POLL_S, remaining))
            command = None
        truncated, count = truncate_text_by_tokens("".join(output), max_output_tokens)
        return truncated.encode("utf-8", errors="replace"), count

    async def _update(self, process_id: int, entry: _Running, output: bytes, count: int | None) -> PtyExecUpdate:
        live: int | None = process_id
        if entry.exit_code is not None:
            async with self._pty_lock:
                self._pty_processes.pop(process_id, None)
                self._reserved_pty_process_ids.discard(process_id)
            live = None
        return PtyExecUpdate(process_id=live, output=output, exit_code=entry.exit_code, original_token_count=count)

    def _prune_pty_processes(self) -> _Running | None:
        if len(self._pty_processes) < PTY_PROCESSES_MAX:
            return None
        meta = [(process_id, entry.last_used, entry.exit_code is not None) for process_id, entry in self._pty_processes.items()]
        target = process_id_to_prune_from_meta(meta)
        if target is None:
            return None
        self._reserved_pty_process_ids.discard(target)
        return self._pty_processes.pop(target, None)

    async def _terminate(self, entries: Sequence[_Running]) -> None:
        """Stops commands left running, by the process ids they wrote down. The API has no call that stops one."""
        pidfiles = [entry.pidfile for entry in entries if entry.exit_code is None]
        if not pidfiles:
            return
        script = 'for f in "$@"; do p=$(cat "$f" 2>/dev/null) && { pkill -TERM -P "$p"; kill -TERM "$p"; }; rm -f "$f"; done 2>/dev/null; exit 0'
        try:
            await self._run(["sh", "-c", script, "sh", *pidfiles], cwd=None, env={}, timeout=30.0)
        except Exception as error:
            logger.warning("Could not stop the commands left running in agental sandbox %s: %s", self.state.sandbox_id, error)

    # ----- files -----

    async def read(self, path: Path, *, user: str | User | None = None) -> io.IOBase:
        if user is not None:
            await self._check_read_with_exec(path, user=user)
        target = await self._validate_path_access(path)
        return io.BytesIO(await self._read_file(coerce_posix_path(target), shown=target))

    async def write(self, path: Path, data: io.IOBase, *, user: str | User | None = None) -> None:
        if user is not None:
            await self._check_write_with_exec(path, user=user)
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=Path(path), actual_type=type(payload).__name__)
        target = await self._validate_path_access(path, for_write=True)
        await self._write_file(coerce_posix_path(target), bytes(payload), shown=target)

    async def persist_workspace(self) -> io.IOBase:
        root = self._workspace_root_path()
        archive = f"{self._staging()}/persist-{uuid.uuid4().hex}.tar"
        skipped = {rel.as_posix() for rel in self._persist_workspace_skip_relpaths()}
        inside = _relative_to(PurePosixPath(self._staging()), coerce_posix_path(root))
        if inside is not None:
            skipped.add(inside)
        excludes = [f"--exclude=./{rel}" for rel in sorted(skipped)]
        script = 'archive=$1; shift; mkdir -p "$(dirname "$archive")" && exec tar cf "$archive" "$@"'
        try:
            made = await self._run(["sh", "-c", script, "sh", archive, "-C", sandbox_path_str(root), *excludes, "."], cwd=None, env={}, timeout=_COMMAND_LIMIT_S)
            if not made.ok():
                raise WorkspaceArchiveReadError(path=root, context=self._context(**_said(made)))
            return io.BytesIO(await self._read_file(PurePosixPath(archive), shown=root))
        finally:
            await self._remove(archive)

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        root = self._workspace_root_path()
        raw = data.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if not isinstance(raw, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=root, actual_type=type(raw).__name__)
        try:
            with tarfile.open(fileobj=io.BytesIO(bytes(raw)), mode="r:*") as archive_file:
                validate_tarfile(archive_file)
        except (UnsafeTarMemberError, tarfile.TarError, OSError) as error:
            raise WorkspaceArchiveWriteError(path=root, cause=error) from error
        archive = f"{self._staging()}/hydrate-{uuid.uuid4().hex}.tar"
        await self._write_file(PurePosixPath(archive), bytes(raw), shown=root)
        script = 'mkdir -p "$2" && tar xf "$1" -C "$2"; rc=$?; rm -f "$1"; exit $rc'
        unpacked = await self._run(["sh", "-c", script, "sh", archive, sandbox_path_str(root)], cwd=None, env={}, timeout=_COMMAND_LIMIT_S)
        if not unpacked.ok():
            raise WorkspaceArchiveWriteError(path=root, context=self._context(**_said(unpacked)))

    async def _read_file(self, target: PurePosixPath, *, shown: Path) -> bytes:
        relative = _relative_to(target, PurePosixPath(self.state.files_root))
        if relative is not None:
            try:
                return await self._sandbox.files.read_bytes(relative)
            except ApiError as error:
                if error.code == "file_not_found":
                    raise WorkspaceReadNotFoundError(path=shown, cause=error) from error
                # Over 10 MB, or not something the files API hands over: read in parts, which says which.
            except AgentalError as error:
                raise WorkspaceArchiveReadError(path=shown, context=self._context(error), cause=error) from error
        return await self._read_in_parts(target, shown=shown)

    async def _read_in_parts(self, target: PurePosixPath, *, shown: Path) -> bytes:
        folder = f"read-{uuid.uuid4().hex}"
        script = f'test -f "$1" || exit {_NOT_FOUND}\nmkdir -p "$2" && cd "$2" && split -b {_PART_BYTES} -a 4 "$1" part- && ls'
        try:
            split = await self._run(["sh", "-c", script, "sh", str(target), f"{self._staging()}/{folder}"], cwd=None, env={}, timeout=_HELD_LIMIT_S)
            if split.exit_code == _NOT_FOUND:
                raise WorkspaceReadNotFoundError(path=shown)
            if not split.ok():
                raise WorkspaceArchiveReadError(path=shown, context=self._context(**_said(split)))
            parts = sorted(name for name in split.stdout.decode("utf-8", errors="replace").split() if name.startswith("part-"))
            data = bytearray()
            for part in parts:
                data += await self._sandbox.files.read_bytes(f"{STAGING}/{folder}/{part}")
            return bytes(data)
        except AgentalError as error:
            raise WorkspaceArchiveReadError(path=shown, context=self._context(error), cause=error) from error
        finally:
            await self._remove(f"{self._staging()}/{folder}")

    async def _write_file(self, target: PurePosixPath, payload: bytes, *, shown: Path) -> None:
        # Always through a copy and cat, so that a file that is there keeps its mode and owner:
        # the files API writes every file as root's, 0644.
        folder = f"write-{uuid.uuid4().hex}"
        try:
            for index, start in enumerate(range(0, len(payload), _PART_BYTES)):
                await self._sandbox.files.write(f"{STAGING}/{folder}/part-{index:06d}", payload[start : start + _PART_BYTES])
        except AgentalError as error:
            await self._remove(f"{self._staging()}/{folder}")
            raise WorkspaceArchiveWriteError(path=shown, context=self._context(error), cause=error) from error
        script = 'mkdir -p "$(dirname "$1")" || exit 1\nif [ -d "$2" ]; then cat "$2"/part-* > "$1"; else : > "$1"; fi\nrc=$?\nrm -rf "$2"\nexit $rc'
        placed = await self._run(["sh", "-c", script, "sh", str(target), f"{self._staging()}/{folder}"], cwd=None, env={}, timeout=_HELD_LIMIT_S)
        if not placed.ok():
            raise WorkspaceArchiveWriteError(path=shown, context=self._context(**_said(placed)))

    async def _remove(self, path: str) -> None:
        try:
            await self._run(["rm", "-rf", "--", path], cwd=None, env={}, timeout=60.0)
        except Exception as error:
            logger.warning("Could not remove %s from agental sandbox %s: %s", path, self.state.sandbox_id, error)

    def _staging(self) -> str:
        return f"{self.state.files_root.rstrip('/')}/{STAGING}"

    # ----- ports -----

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        try:
            route = next((each for each in await self._sandbox.routes() if each.port == port), None)
            if route is None:
                route = await self._sandbox.expose(port, title=f"openai-agents-{port}", public=True)
        except AgentalError as error:
            raise ExposedPortUnavailableError(
                port=port, exposed_ports=self.state.exposed_ports, reason="backend_unavailable", context=self._context(error), cause=error
            ) from error
        address = urlsplit(route.url or "")
        if not address.hostname:
            # A route that is not public, or a deployment with public addresses turned off, has no address to give.
            raise ExposedPortUnavailableError(
                port=port, exposed_ports=self.state.exposed_ports, reason="backend_unavailable", context=self._context(reason="no_public_address")
            )
        tls = address.scheme == "https"
        return ExposedPortEndpoint(host=address.hostname, port=address.port or (443 if tls else 80), tls=tls)

    def _context(self, error: AgentalError | None = None, /, **more: object) -> dict[str, object]:
        context: dict[str, object] = {"backend": "agental", "sandbox_id": self.state.sandbox_id}
        if isinstance(error, ApiError):
            context.update(status=error.status, code=error.code, message=error.message)
        elif error is not None:
            context.update(message=str(error))
        context.update(more)
        return context


class AgentalSandboxClient(BaseSandboxClient[AgentalSandboxClientOptions | None]):
    """Makes and resumes sessions in agental sandboxes.

    api_key is read from AGENTAL_API_KEY when left out, and base_url from AGENTAL_API_URL,
    then https://agental.io. With neither given, the key the agental command signed in with
    on this computer is used. agental is a client of the agental SDK to use in their place.
    files_root is where the files API reads and writes on a sandbox: leave it, except against
    an API on the local driver, whose sandboxes are folders.
    """

    backend_id = "agental"
    supports_default_options = True

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        agental: AsyncAgental | None = None,
        files_root: str = FILES_ROOT,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        super().__init__()
        try:
            self._agental = agental or AsyncAgental(api_key, base_url=base_url)
        except AgentalError as error:
            raise ConfigurationError(message=str(error), error_code=ErrorCode.SANDBOX_CONFIG_INVALID, op="start", context={"backend": "agental"}) from error
        # A client the caller passed in is theirs to close.
        self._owns_agental = agental is None
        self._files_root = files_root.rstrip("/") or "/"
        self._instrumentation = instrumentation or Instrumentation()
        self._dependencies = dependencies

    @property
    def agental(self) -> AsyncAgental:
        """The agental SDK client the sessions use."""
        return self._agental

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: AgentalSandboxClientOptions | None = None,
    ) -> SandboxSession:
        options = options or AgentalSandboxClientOptions()
        manifest = self._validate_manifest_for_create(_with_root(manifest, self._files_root))
        root = posix_path_as_path(coerce_posix_path(manifest.root))
        session_id = uuid.uuid4()
        try:
            if options.sandbox_id is not None:
                given = [
                    name
                    for name, value in (
                        ("name", options.name),
                        ("template", options.template),
                        ("size", options.size),
                        ("snapshot_id", options.snapshot_id),
                        ("allow", options.allow),
                        ("ttl_seconds", options.ttl_seconds),
                        ("region", options.region),
                        ("ssh_keys", options.ssh_keys),
                    )
                    if value is not None
                ]
                if given:
                    raise ConfigurationError(
                        message=f"{', '.join(given)} {'is' if len(given) == 1 else 'are'} for a new sandbox, and sandbox_id names one that is there",
                        error_code=ErrorCode.SANDBOX_CONFIG_INVALID,
                        op="start",
                        context={"backend": "agental", "sandbox_id": options.sandbox_id},
                    )
                sandbox = await self._agental.sandboxes.get(options.sandbox_id)
            else:
                sandbox = await _make_sandbox(
                    self._agental,
                    root=root,
                    name=options.name,
                    template=options.template,
                    size=options.size,
                    snapshot_id=options.snapshot_id,
                    allow=options.allow,
                    ttl_seconds=options.ttl_seconds,
                    region=options.region,
                    ssh_keys=options.ssh_keys,
                    max_burst_fee_micros=options.max_burst_fee_micros,
                    max_burst_wait=options.max_burst_wait,
                )
        except AgentalError as error:
            raise WorkspaceStartError(path=root, context=_api_context(error, options.sandbox_id), cause=error) from error
        state = AgentalSandboxSessionState(
            session_id=session_id,
            manifest=manifest,
            snapshot=resolve_snapshot(snapshot, str(session_id)),
            exposed_ports=options.exposed_ports,
            sandbox_id=sandbox.id,
            created_by_us=options.sandbox_id is None,
            files_root=self._files_root,
            name=options.name,
            template=options.template,
            size=options.size,
            snapshot_id=options.snapshot_id,
            allow=options.allow,
            ttl_seconds=options.ttl_seconds,
            region=options.region,
            ssh_keys=options.ssh_keys,
            max_burst_fee_micros=options.max_burst_fee_micros,
            max_burst_wait=options.max_burst_wait,
            stop_on_exit=options.stop_on_exit,
        )
        inner = AgentalSandboxSession(state=state, agental=self._agental, sandbox=sandbox)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, AgentalSandboxSession):
            raise TypeError("AgentalSandboxClient.delete expects a session it made")
        await inner.shutdown()
        return session

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        if not isinstance(state, AgentalSandboxSessionState):
            raise TypeError("AgentalSandboxClient.resume expects an AgentalSandboxSessionState")
        root = posix_path_as_path(coerce_posix_path(state.manifest.root))
        sandbox: AsyncSandbox | None = None
        try:
            found = await self._agental.sandboxes.get(state.sandbox_id)
            if found.status != "deleted":
                sandbox = found
        except ApiError as error:
            if error.status != 404:
                raise WorkspaceStartError(path=root, context=_api_context(error, state.sandbox_id, reason="reattach_failed"), cause=error) from error
        except AgentalError as error:
            raise WorkspaceStartError(path=root, context=_api_context(error, state.sandbox_id, reason="reattach_failed"), cause=error) from error
        if sandbox is not None:
            inner = AgentalSandboxSession(state=state, agental=self._agental, sandbox=sandbox)
            inner._set_start_state_preserved(True)
            return self._wrap_session(inner, instrumentation=self._instrumentation)
        if not state.created_by_us:
            # One that was given by its id is not quietly replaced by another.
            raise WorkspaceStartError(path=root, context={"backend": "agental", "sandbox_id": state.sandbox_id, "reason": "sandbox_not_found"})
        # The sandbox this session made is gone: a new one is made, and start() puts the workspace in it
        # from the session's snapshot, or from the manifest when there is none.
        try:
            sandbox = await _make_sandbox(
                self._agental,
                root=root,
                name=None,
                template=state.template,
                size=state.size,
                snapshot_id=state.snapshot_id,
                allow=state.allow,
                ttl_seconds=state.ttl_seconds,
                region=state.region,
                ssh_keys=state.ssh_keys,
                max_burst_fee_micros=state.max_burst_fee_micros,
                max_burst_wait=state.max_burst_wait,
            )
        except AgentalError as error:
            raise WorkspaceStartError(path=root, context=_api_context(error, state.sandbox_id, reason="replace_failed"), cause=error) from error
        state.sandbox_id = sandbox.id
        state.workspace_root_ready = False
        inner = AgentalSandboxSession(state=state, agental=self._agental, sandbox=sandbox)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, AgentalSandboxSessionState)

    async def aclose(self) -> None:
        """Closes the agental SDK client, when this client made it."""
        if self._owns_agental:
            await self._agental.aclose()

    async def __aenter__(self) -> AgentalSandboxClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


async def _make_sandbox(
    agental: AsyncAgental,
    *,
    root: Path,
    name: str | None,
    template: str | None,
    size: Size | None,
    snapshot_id: str | None,
    allow: tuple[str, ...] | None,
    ttl_seconds: int | None,
    region: str | tuple[str, ...] | None,
    ssh_keys: tuple[str, ...] | None,
    max_burst_fee_micros: int | None,
    max_burst_wait: float | None,
) -> AsyncSandbox:
    template_id: str | None = None
    if template is not None:
        found = next((each for each in await agental.templates.list() if template in (each.id, each.name)), None)
        if found is None:
            raise WorkspaceStartError(path=root, context={"backend": "agental", "template": template, "reason": "template_not_found"})
        template_id = found.id
    # Only what was given is sent, so the API's defaults stand for the rest. The time limit goes with the create: set
    # after it, a failure would leave a sandbox that nobody deletes running. Left out, the template's start says.
    given: dict[str, Any] = {}
    if ttl_seconds is not None:
        given["ttl_seconds"] = ttl_seconds
    if region is not None:
        given["region"] = region
    if ssh_keys is not None:
        given["ssh_keys"] = list(ssh_keys)
    if max_burst_fee_micros is not None:
        given["max_burst_fee_micros"] = max_burst_fee_micros
    if max_burst_wait is not None:
        given["max_burst_wait"] = max_burst_wait
    return await agental.sandboxes.create(
        name or f"openai-agents-{uuid.uuid4().hex[:12]}",
        template=template_id,
        size=size,
        snapshot_id=snapshot_id,
        allow=list(allow) if allow is not None else None,
        **given,
    )


def command_script(argv: Sequence[str], *, cwd: str | None, env: Mapping[str, str], pidfile: str | None = None) -> str:
    """The sh that runs a command in its folder with its variables, and writes down its process id when asked."""
    lines: list[str] = []
    if pidfile is not None:
        folder = pidfile.rsplit("/", 1)[0] or "/"
        lines.append(f"mkdir -p {shlex.quote(folder)} && echo $$ > {shlex.quote(pidfile)}")
    if cwd is not None:
        lines.append(f"cd {shlex.quote(cwd)} || exit 1")
    lines.extend(f"export {name}={shlex.quote(value)}" for name, value in env.items())
    lines.append(f"exec {shlex.join(argv)}")
    return "\n".join(lines) + "\n"


def wire_command(argv: Sequence[str], *, cwd: str | None, env: Mapping[str, str], pidfile: str | None = None) -> list[str] | None:
    """The arguments to send the API for a command, or None when it is too long to go as arguments."""
    if cwd is None and not env and pidfile is None and _fits(argv):
        return list(argv)
    script = command_script(argv, cwd=cwd, env=env, pidfile=pidfile)
    if _units(script) <= _ARG_LIMIT:
        return ["sh", "-c", script]
    # In pieces, which sh joins with nothing between them, as $* is with IFS empty.
    pieces = _pieces(script)
    if len(pieces) + 4 <= _ARG_COUNT:
        return ["sh", "-c", 'IFS=; eval "$*"', "sh", *pieces]
    return None


def _fits(argv: Sequence[str]) -> bool:
    return 0 < len(argv) <= _ARG_COUNT and all(0 < _units(each) <= _ARG_LIMIT for each in argv)


def _units(text: str) -> int:
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def _pieces(text: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for character in text:
        width = 2 if ord(character) > 0xFFFF else 1
        if size + width > _ARG_LIMIT:
            pieces.append("".join(current))
            current, size = [], 0
        current.append(character)
        size += width
    if current:
        pieces.append("".join(current))
    return pieces


def _newer(seen: str, now: str) -> str:
    """What a command's log has that it did not have last time, when the API keeps only its last 64 KB."""
    if now.startswith(seen):
        return now[len(seen) :]
    anchor = seen[-512:]
    at = now.rfind(anchor) if anchor else -1
    if at >= 0:
        return now[at + len(anchor) :]
    return _DROPPED + now


def _relative_to(path: PurePosixPath, root: PurePosixPath) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    text = relative.as_posix()
    return None if text in ("", ".") else text


def _with_root(manifest: Manifest | None, files_root: str) -> Manifest:
    """The manifest with its root in the files API's folder, when it was left at the SDK's default."""
    root = f"{files_root.rstrip('/')}/workspace"
    if manifest is None:
        return Manifest(root=root)
    if manifest.root == _DEFAULT_MANIFEST_ROOT:
        return manifest.model_copy(update={"root": root})
    return manifest


def _said(result: ExecResult) -> dict[str, object]:
    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout.decode("utf-8", errors="replace")[-4096:],
        "stderr": result.stderr.decode("utf-8", errors="replace")[-4096:],
    }


def _retryable(error: AgentalError) -> bool | None:
    if isinstance(error, ApiError):
        return error.status == 503 or error.code in ("sandbox_moved", "sandbox_migrating")
    return None


def _api_context(error: AgentalError, sandbox_id: str | None, /, **more: object) -> dict[str, object]:
    context: dict[str, object] = {"backend": "agental"}
    if sandbox_id is not None:
        context["sandbox_id"] = sandbox_id
    if isinstance(error, ApiError):
        context.update(status=error.status, code=error.code, message=error.message)
    else:
        context["message"] = str(error)
    context.update(more)
    return context


__all__ = [
    "FILES_ROOT",
    "AgentalSandboxClient",
    "AgentalSandboxClientOptions",
    "AgentalSandboxSession",
    "AgentalSandboxSessionState",
    "command_script",
    "wire_command",
]
