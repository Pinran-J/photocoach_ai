"""
Quick test client for the PhotoCoach MCP server.
Run this while mcp_server.py is running in another terminal.

Usage:
    python test_mcp_client.py
"""

import asyncio
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SERVER_URL = "http://16.176.199.85:8000/mcp"  # deployed on AWS ECS; use http://localhost:8000/mcp for local testing


async def main():
    print(f"Connecting to {SERVER_URL}...\n")

    async with streamablehttp_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List available tools
            tools = await session.list_tools()
            print("=== Available MCP tools ===")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:80]}")

            # 2. Test retrieval tool
            print("\n=== Testing retrieve_photography_tips ===")
            result = await session.call_tool(
                "retrieve_photography_tips",
                {"query": "how to use rule of thirds in composition"},
            )
            print(result.content[0].text[:600])

            # 3. Test aesthetic scorer with a public image URL
            print("\n=== Testing score_aesthetic ===")
            test_url = "https://picsum.photos/seed/photography/640/480"  # permissive CDN, always works
            result = await session.call_tool(
                "score_aesthetic",
                {"image_url": test_url},
            )
            print(result.content[0].text)

            print("\nAll tools working correctly.")


asyncio.run(main())
