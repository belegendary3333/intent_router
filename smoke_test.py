"""Test a running intent-router; only intent inference, no robot actions."""

import argparse
import json
import urllib.request


def request(url: str, text: str | None = None) -> dict:
    data = None if text is None else json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8890")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    print(json.dumps(request(f"{base_url}/health"), ensure_ascii=False, indent=2))
    for text, expected in (
        ("陪我聊聊天", "chat"),
        ("帮我擦桌子", "skill"),
        ("小诺，给我唱首歌", "singing"),
        ("跟着音乐跳舞", "dancing"),
        ("小诺进入专注模式", "focus"),
        ("我梦见自己在唱歌", "chat"),
    ):
        result = request(f"{base_url}/route", text)
        print(json.dumps(result, ensure_ascii=False))
        if result.get("mode") != expected:
            raise SystemExit(f"unexpected route: {text!r}, expected={expected}, actual={result.get('mode')}")
    print("[OK] Five live BGE routing checks passed.")


if __name__ == "__main__":
    main()
