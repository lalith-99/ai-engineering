"""
Day 8: MCP (Model Context Protocol) server.

A simple MCP server that exposes tools for querying a mock project database.
MCP lets any LLM client (Claude Desktop, Cursor, etc.) connect and use these tools.

Requires: pip install mcp

Usage:
    python mcp_server.py

Then connect from an MCP client by pointing it at this script.
"""

import json
from datetime import datetime

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise SystemExit("MCP SDK not installed. Run: pip install mcp")


mcp = FastMCP("project-tracker")

# Mock project data
PROJECTS = {
    "nimbus": {
        "name": "Nimbus",
        "status": "active",
        "language": "Go",
        "description": "Notification gateway with SQS, SNS, and multi-channel delivery",
        "last_deploy": "2026-02-15",
        "health": "healthy",
        "open_issues": 3,
    },
    "echostream": {
        "name": "EchoStream",
        "status": "active",
        "language": "Go",
        "description": "Real-time messaging platform with WebSocket support",
        "last_deploy": "2026-02-10",
        "health": "healthy",
        "open_issues": 7,
    },
    "filestream": {
        "name": "FileStream",
        "status": "maintenance",
        "language": "Go",
        "description": "File transfer service with S3, SFTP, and Kafka integration",
        "last_deploy": "2026-01-28",
        "health": "degraded",
        "open_issues": 12,
    },
}


@mcp.tool()
def list_projects() -> str:
    """List all projects with their current status."""
    rows = []
    for key, p in PROJECTS.items():
        rows.append(f"- {p['name']} ({p['language']}) — {p['status']}, {p['open_issues']} open issues")
    return "\n".join(rows)


@mcp.tool()
def get_project_details(project_name: str) -> str:
    """Get detailed info about a specific project.

    Args:
        project_name: The project name (nimbus, echostream, filestream)
    """
    key = project_name.lower().replace(" ", "")
    project = PROJECTS.get(key)
    if not project:
        return f"Project '{project_name}' not found. Available: {', '.join(PROJECTS.keys())}"
    return json.dumps(project, indent=2)


@mcp.tool()
def get_project_health(project_name: str) -> str:
    """Check the health status and last deployment date of a project.

    Args:
        project_name: The project name
    """
    key = project_name.lower().replace(" ", "")
    project = PROJECTS.get(key)
    if not project:
        return f"Project '{project_name}' not found."

    last_deploy = datetime.strptime(project["last_deploy"], "%Y-%m-%d")
    days_ago = (datetime.now() - last_deploy).days

    status = f"Health: {project['health']}\n"
    status += f"Last deploy: {project['last_deploy']} ({days_ago} days ago)\n"
    if days_ago > 14:
        status += "Warning: no deployment in over 2 weeks"
    return status


@mcp.resource("projects://summary")
def project_summary() -> str:
    """Summary of all projects as an MCP resource."""
    total_issues = sum(p["open_issues"] for p in PROJECTS.values())
    active = sum(1 for p in PROJECTS.values() if p["status"] == "active")
    return (
        f"Total projects: {len(PROJECTS)}\n"
        f"Active: {active}\n"
        f"Total open issues: {total_issues}\n"
        f"As of: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


if __name__ == "__main__":
    print("Starting MCP server: project-tracker")
    print("Connect from Claude Desktop or any MCP client.")
    mcp.run()
