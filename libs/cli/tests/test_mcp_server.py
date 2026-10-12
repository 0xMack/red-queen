"""The MCP server (redqueen_cli.mcp_server) as an agent sees it: an MCP client session connected in-process, its tools
listed and called against the real API (the backend's TestClient)."""

import asyncio
from typing import Any

import pytest

pytest.importorskip("mcp.server.mcpserver")


@pytest.fixture
def agent(cli):
    """`agent(tool, **arguments)` calls a tool over an MCP session; `agent.tools()` lists them."""
    from backend.main import app as api
    from fastapi.testclient import TestClient
    from mcp import Client as McpClient

    from redqueen_cli.client import Client
    from redqueen_cli.mcp_server import build_server

    server = build_server(Client(transport=TestClient(api)))

    async def call(tool: str, arguments: dict[str, Any]):
        async with McpClient(server) as session:
            return await session.call_tool(tool, arguments)

    async def tools():
        async with McpClient(server) as session:
            return (await session.list_tools()).tools

    listed = {t.name: t for t in asyncio.run(tools())}

    def invoke(tool: str, /, **arguments: Any) -> Any:
        result = asyncio.run(call(tool, arguments))
        if result.is_error:
            raise ToolError(result.content[0].text)
        # structured output is an object: a tool returning anything else has it wrapped as {"result": ...}
        wrapped = set((listed[tool].output_schema or {}).get("properties", {})) == {"result"}
        return result.structured_content["result"] if wrapped else result.structured_content

    invoke.tools = lambda: listed
    invoke.cli = cli
    return invoke


class ToolError(Exception):
    pass


def test_every_tool_is_listed_with_its_hints(agent):
    tools = agent.tools()
    assert {"schemas", "list_runs", "train", "submit_job", "get_job", "job_logs", "cancel_job", "leaderboard",
            "submit_experiment", "experiment_status", "submit_spec_file", "submit_schedule"} <= set(tools)  # fmt: skip
    assert tools["list_runs"].annotations.read_only_hint and not tools["train"].annotations.read_only_hint
    assert tools["cancel_job"].annotations.destructive_hint
    assert "budget" in tools["train"].input_schema["properties"]


def test_submit_then_poll_then_cancel(agent):
    job = agent("train", algorithm="gp", game="symbolic-regression", budget={"generations": 2}, seed=4)
    assert job["status"] == "queued" and job["spec"]["seed"] == 4
    assert agent("get_job", job=job["job_id"][:8])["status"] == "queued"
    assert agent("job_logs", job=job["job_id"])["offset"] == 0
    assert agent("cancel_job", job=job["job_id"])["status"] == "cancelled"
    assert [j["job_id"] for j in agent("list_jobs", status="cancelled")] == [job["job_id"]]


def test_a_refused_spec_is_a_tool_error_with_the_reason(agent):
    with pytest.raises(ToolError, match="nope"):
        agent("train", algorithm="nope", game="snake", budget={"generations": 1})


def test_runs_schemas_and_spec_files(agent, tmp_path):
    run = agent.cli.registry.create_run({"game": "checkers", "representation": "td_lambda"})
    assert [r["run_id"] for r in agent("list_runs", game="checkers")] == [run]
    assert agent("get_run", run=run[:6])["run_id"] == run
    assert "td_lambda" in agent("schemas")["algorithms"]

    spec = tmp_path / "probe.yaml"
    spec.write_bytes(b"kind: experiment\nname: probe\nseeds: [0]\ndefaults: {game: snake, algorithm: neat, "
                     b"budget: {generations: 2}}\narms: {a: {}}\n")  # fmt: skip
    matrix = agent("submit_spec_file", path=str(spec), overrides=["seeds=[0, 1]"])
    assert matrix["name"] == "probe" and matrix["total"] == 2
    assert agent("experiment_status", name="probe")["done"] == 0
