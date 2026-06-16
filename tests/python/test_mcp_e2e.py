"""End-to-end tests for the advertised MCP launcher (`strands-shell --mcp`).

These exercise the exact path the README points MCP clients at:

    {"command": "uvx", "args": ["strands-shell", "--mcp", "--config", "..."]}

i.e. the `strands-shell` console script (wired in pyproject.toml's
`[project.scripts]` to `strands_shell._native:cli_main`), running the stdio
MCP server over real OS pipes — not the in-process Rust `serve_io` harness that
`tests/mcp_integration.rs` covers. This is the only layer that proves the
launcher, argv handling, `--config` parsing, and newline-delimited JSON-RPC
framing all work together as a published wheel would run them.

The protocol is JSON-RPC 2.0, one JSON object per line, over stdin/stdout
(see `src/mcp.rs::serve`). We write every request, close stdin, then read the
responses the server flushed before it hit EOF and exited.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest


def _launcher_cmd():
    """Resolve the command that runs the MCP server, hermetically.

    The advertised entry point is the `strands-shell` console script that
    `pip install` / `uvx` generate from pyproject's `[project.scripts]`
    (`strands_shell._native:cli_main`). We must drive the script that belongs
    to *this* interpreter's environment — never `shutil.which("strands-shell")`,
    which can return a launcher from an unrelated venv elsewhere on PATH and
    silently test a different build.

    So: prefer the console script sitting next to `sys.executable` (same env as
    the freshly built/installed extension). `maturin develop` doesn't always
    generate that shim, so fall back to invoking `cli_main` through this same
    interpreter — the generated shim is itself just
    `sys.exit(strands_shell._native.cli_main())`, so this drives the identical
    Rust code path (argv -> cli::run -> mcp::serve) against the same build.
    """
    script = os.path.join(os.path.dirname(sys.executable), "strands-shell")
    if os.path.isfile(script) and os.access(script, os.X_OK):
        return [script]
    return [
        sys.executable,
        "-c",
        "import sys; from strands_shell._native import cli_main; "
        "sys.exit(cli_main())",
        # argv[0] is the -c program; the launcher name the user typed is
        # irrelevant to the server — only the flags after it matter.
    ]


def mcp_exchange(requests, *, config=None, timeout=30):
    """Spawn the launcher in `--mcp` mode, feed `requests`, return responses.

    `requests` is a list of JSON-RPC objects. Returns the parsed JSON objects
    the server wrote back (notifications produce no response, so the count is
    only the requests that carried an `id`).
    """
    cmd = list(_launcher_cmd())
    cmd.append("--mcp")
    if config is not None:
        cmd += ["--config", config]

    stdin_data = "".join(json.dumps(r) + "\n" for r in requests)
    proc = subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    # The server exits 0 on clean EOF; a non-zero code means the launcher
    # itself failed (bad argv, config error) — surface stderr to debug.
    assert proc.returncode == 0, (
        f"launcher exited {proc.returncode}; stderr:\n{proc.stderr}"
    )
    return [
        json.loads(line)
        for line in proc.stdout.splitlines()
        if line.strip()
    ]


def _init(id=1):
    return {
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest-e2e", "version": "0.1"},
        },
    }


def _initialized():
    return {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _tool_call(id, name, arguments):
    return {
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


@pytest.fixture
def host_dir():
    path = tempfile.mkdtemp(prefix="strands-shell-mcp-e2e-")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ── handshake ────────────────────────────────────────────────────────


def test_initialize_reports_server_identity():
    responses = mcp_exchange([_init(1)])
    assert len(responses) == 1
    r = responses[0]
    assert r["jsonrpc"] == "2.0"
    assert r["id"] == 1
    assert r["result"]["protocolVersion"] == "2024-11-05"
    assert r["result"]["capabilities"]["tools"] is not None
    assert r["result"]["serverInfo"]["name"] == "strands-shell"


def test_tools_list_exposes_advertised_tools():
    responses = mcp_exchange(
        [
            _init(1),
            _initialized(),
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
    )
    tools = responses[-1]["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"shell", "read_file", "write_file", "list_dir"}


# ── shell tool ───────────────────────────────────────────────────────


def test_shell_echo():
    responses = mcp_exchange(
        [_init(1), _initialized(), _tool_call(2, "shell", {"command": "echo hello"})]
    )
    text = responses[-1]["result"]["content"][0]["text"]
    assert text.strip() == "hello"


def test_shell_pipeline():
    responses = mcp_exchange(
        [
            _init(1),
            _initialized(),
            _tool_call(2, "shell", {"command": "echo hello | tr a-z A-Z"}),
        ]
    )
    text = responses[-1]["result"]["content"][0]["text"]
    assert text.strip() == "HELLO"


def test_shell_nonzero_exit_code_in_metadata():
    responses = mcp_exchange(
        [_init(1), _initialized(), _tool_call(2, "shell", {"command": "false"})]
    )
    assert responses[-1]["result"]["metadata"]["exit_code"] == 1


# ── state persists across calls on one stdio session ─────────────────


def test_state_persists_across_tool_calls():
    """A single launcher process keeps shell state between tool calls —
    the property that makes a long-lived MCP session useful."""
    responses = mcp_exchange(
        [
            _init(1),
            _initialized(),
            _tool_call(2, "shell", {"command": "export GREETING=hi"}),
            _tool_call(3, "shell", {"command": "echo $GREETING"}),
        ]
    )
    text = responses[-1]["result"]["content"][0]["text"]
    assert text.strip() == "hi"


# ── write/read round trip ────────────────────────────────────────────


def test_write_then_read_file():
    responses = mcp_exchange(
        [
            _init(1),
            _initialized(),
            _tool_call(
                2,
                "write_file",
                {"file_path": "/tmp/e2e.txt", "content": "hello world"},
            ),
            _tool_call(3, "read_file", {"file_path": "/tmp/e2e.txt"}),
        ]
    )
    write_text = responses[-2]["result"]["content"][0]["text"]
    assert "11 bytes" in write_text
    read_text = responses[-1]["result"]["content"][0]["text"]
    assert "hello world" in read_text


def test_read_nonexistent_file_is_error():
    responses = mcp_exchange(
        [
            _init(1),
            _initialized(),
            _tool_call(2, "read_file", {"file_path": "/nonexistent.txt"}),
        ]
    )
    assert responses[-1]["result"]["isError"] is True


# ── the advertised --config path (bind mounts) ───────────────────────


def test_config_bind_exposes_host_file(host_dir):
    """The README's invocation passes `--config sandbox.toml` declaring bind
    mounts. Prove a host file under a `[[bind]]` is reachable through the MCP
    server launched exactly that way."""
    with open(os.path.join(host_dir, "data.txt"), "w") as f:
        f.write("from-host")

    config_path = os.path.join(host_dir, "sandbox.toml")
    with open(config_path, "w") as f:
        f.write(
            "[[bind]]\n"
            'mode = "direct"\n'
            f'source = "{host_dir}"\n'
            'destination = "/work"\n'
        )

    responses = mcp_exchange(
        [
            _init(1),
            _initialized(),
            _tool_call(2, "read_file", {"file_path": "/work/data.txt"}),
        ],
        config=config_path,
    )
    text = responses[-1]["result"]["content"][0]["text"]
    assert "from-host" in text
