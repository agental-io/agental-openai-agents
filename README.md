# agental-openai-agents

Runs an [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/sandbox_agents/) `SandboxAgent` in an [agental.io](https://agental.io) sandbox: a Linux machine of your own. The agent's commands and file edits happen in the sandbox. Your code and the model's calls stay where they are.

```bash
pip install agental-openai-agents
```

Python 3.10 or later. It depends on `agental`, the agental SDK, and on `openai-agents` 0.24. Each release of this package supports the Agents SDK versions it was tested with, and allows no others, since that SDK's sandbox interface changes often. This package is made by agental, not by OpenAI.

## Start

Make an API key in the agental console and put it in `AGENTAL_API_KEY`, next to `OPENAI_API_KEY`.

```python
import asyncio

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agental_openai_agents import AgentalSandboxClient, AgentalSandboxClientOptions

agent = SandboxAgent(name="Fixer", instructions="Run the tests in the workspace and fix what fails.")


async def main() -> None:
    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=AgentalSandboxClient(),  # AGENTAL_API_KEY, or the agental command's sign-in
            options=AgentalSandboxClientOptions(size="small"),
        ),
    )
    result = await Runner.run(agent, "Make the tests pass.", run_config=run_config)
    print(result.final_output)


asyncio.run(main())
```

Moving from another sandbox client changes only `client=` and `options=`. The agent stays as it is.

## What happens in a run

1. The client makes a sandbox from your account's default template, or the one you name.
2. The Agents SDK writes the manifest's files into the workspace, `/opt/agental/workspace` unless the manifest names another root.
3. Each command the model runs goes to the sandbox, in the workspace and with the manifest's variables, and its output goes back to the model.
4. When the run ends, the sandbox is deleted. It is billed for the time it ran.

A sandbox on agental.io has no time limit unless you or its template give it one. If your process dies before the run ends, the sandbox goes on running and is billed until you stop it: set `ttl_seconds` as a safety net.

## Options

| Option | What it does |
|---|---|
| `size` | `tiny`, `small`, `medium` or `large`. Left out, the template's. |
| `template` | The template to make the sandbox from and hold it to, by id or name. Left out, the account's default. |
| `name` | The sandbox's name. Made up when left out. |
| `snapshot_id` | Make the sandbox from one of your snapshots. |
| `allow` | Addresses the sandbox may reach on top of its template's outbound rule. |
| `ttl_seconds` | Seconds it runs after a start before the API stops it. Left out, the template's start says; with none there, the deployment's default applies, which on agental.io is none, so it runs until it is stopped. |
| `region` | The region to make it in, as `"eu"`, or several to try in order, as `("eu", "us")`: it is made in the first with room. Left out, the default region. |
| `ssh_keys` | OpenSSH public key lines let in as root, on top of the keys its template lets in. They do not turn SSH on: its template or the agental SDK's `ssh.on()` does. |
| `sandbox_id` | Run in a sandbox you already have. It is not deleted at the end, and the options above, which are for a new one, cannot be given with it. |
| `max_burst_fee_micros` | The most a create, or the start of a stopped sandbox, may pay in burst fees, in millionths of a US dollar. `0` refuses any fee. Left out, what the account's monthly burst budget allows. |
| `max_burst_wait` | Seconds a create or start refused for its burst fee may wait, when the API names a time it would cost nothing by, to be tried once more. |
| `stop_on_exit` | Stop the sandbox at the end in place of deleting it, so a later run can pick it up with its files. |
| `exposed_ports` | Ports that get a public address when the agent asks for one. |

## Picking a run up later

With `stop_on_exit=True`, the sandbox is stopped when the run ends, and its disk is kept at 5% of the running price. A run resumed from its state, after an approval for example, starts that sandbox again with the workspace as it was. Without it, the sandbox is deleted, and a resumed run gets a new one, with the workspace put back from the run's snapshot when you gave `SandboxRunConfig` one.

A stopped sandbox stays until you delete it, in the console or with `agental.sandboxes.get(id).delete()`.

## Ports

A port in `exposed_ports` gets a public address the first time the agent asks for it. Anyone with the address can open it while the sandbox runs. What listens must listen on `0.0.0.0`.

## Limits

- A command runs for at most ten minutes. Commands the model runs are left running and read as they go, so it can start a build and come back to it.
- Nothing can be typed into a command once it runs, and there is no terminal. Give a command its input with a pipe or a here-document.
- Files outside `/opt/agental` are read and written through a copy there. It is made and taken away by the same call.
- Mounting cloud storage buckets is not supported yet.

## Development

```bash
pip install -e . pytest mypy
mypy
pytest
```

The tests are in `tests`. The live tests, `test_live.py`, run only inside the agental repository, as they start its API on the local driver: they are not in this public repository. A sandbox on the local driver is a folder, so they give the client `files_root`, the folder the files API reads and writes, which is `/opt/agental` on every real sandbox. In the agental repository, install the agental SDK, agental-sdk, from its folder first, `pip install -e ../python`.
