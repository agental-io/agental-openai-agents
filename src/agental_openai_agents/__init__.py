"""agental sandboxes for the OpenAI Agents SDK.

    from agents import Runner
    from agents.run import RunConfig
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from agental_openai_agents import AgentalSandboxClient, AgentalSandboxClientOptions

    agent = SandboxAgent(name="Fixer", instructions="Run the tests in the workspace and fix what fails.")
    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=AgentalSandboxClient(),  # reads AGENTAL_API_KEY
            options=AgentalSandboxClientOptions(size="small"),
        ),
    )
    result = await Runner.run(agent, "Make the tests pass.", run_config=run_config)
"""

from importlib.metadata import PackageNotFoundError, version as _version

from .sandbox import (
    FILES_ROOT,
    AgentalSandboxClient,
    AgentalSandboxClientOptions,
    AgentalSandboxSession,
    AgentalSandboxSessionState,
)

#: The installed release, read from the package's metadata so that pyproject.toml is the one place it is written.
try:
    __version__ = _version("agental-openai-agents")
except PackageNotFoundError:  # Imported from a source folder that was never installed.
    __version__ = "0.0.0"

__all__ = [
    "FILES_ROOT",
    "AgentalSandboxClient",
    "AgentalSandboxClientOptions",
    "AgentalSandboxSession",
    "AgentalSandboxSessionState",
]
