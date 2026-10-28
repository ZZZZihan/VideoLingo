"""Write only this project's Codex MCP configuration, preserving existing files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tomllib


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root,
                        help="Codex project directory; use .. when opening the parent VideoLingo repository")
    options = parser.parse_args()
    project_root = options.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise SystemExit("Codex 项目目录不存在。")
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise SystemExit("请先创建并安装项目 .venv；参见 README.md。")
    config = project_root / ".codex" / "config.toml"
    config.parent.mkdir(exist_ok=True)
    previous = config.read_text(encoding="utf-8") if config.exists() else ""
    current = tomllib.loads(previous)
    if "videolingo" in current.get("mcp_servers", {}):
        raise SystemExit("项目已有 videolingo MCP 配置，已保留；如需修改请直接编辑该条目。")
    paths = [str(Path(p).parent) for name in ("ffmpeg", "ffprobe") if (p := shutil.which(name))]
    paths += os.environ.get("PATH", os.defpath).split(os.pathsep)
    environment_path = os.pathsep.join(dict.fromkeys(paths))
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    addition = f'''\n[mcp_servers.videolingo]
command = {quote(python)}
args = ["-m", "videolingo_mcp"]
cwd = {quote(root)}
startup_timeout_sec = 20
tool_timeout_sec = 60

[mcp_servers.videolingo.env]
VIDEOLINGO_DATA_DIR = {quote(root / '.videolingo')}
PATH = {quote(environment_path)}
'''
    combined = previous.rstrip() + "\n" + addition
    tomllib.loads(combined)
    config.write_text(combined, encoding="utf-8")
    print(f"已写入项目配置：{config}")
    print("在 Codex 中重新连接本项目的 MCP，或重新打开本项目聊天；不更改全局配置。")


if __name__ == "__main__":
    main()
