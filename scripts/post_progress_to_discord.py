#!/usr/bin/env python3
"""Post doc2graph loop progress to a Discord channel using environment credentials."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    channel_ids = os.environ.get("DISCORD_CHANNEL_IDS", "")
    channel_id = next((item.strip() for item in channel_ids.split(",") if item.strip()), "")
    message = sys.stdin.read().strip()
    if not token or not channel_id or not message:
        return 0

    request = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{channel_id}/messages",
        data=json.dumps({"content": message[:1900]}).encode("utf-8"),
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "doc2graph-loop",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError):
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
