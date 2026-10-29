# VideoLingo MCP

把本地视频交给 Codex，完成语音识别、字幕翻译和视频导出。识别和视频处理在本机运行，翻译使用当前 Codex 对话的能力，无需另配翻译 API 密钥。

本模块是独立 MCP 服务，可作为 VideoLingo 仓库中的 `mcp/` 子项目使用，拥有单独的依赖与任务数据，不调用现有 Streamlit 应用的内部流水线。

## 第一版功能

- 本地视频语音识别：faster-whisper，CPU int8，可指定语言或自动检测。
- 导入已有、经检查的逐段转录，避免重复识别。
- Codex 分页读取转录并分批提交中文翻译，保留原时间戳与片段 ID。
- 导出中文或双语 SRT / WebVTT；未翻译完整时拒绝导出。
- 导出 H.264/AAC MP4，保留全部输入音轨，内嵌默认启用的可切换字幕轨。`burn_in` 可烧录硬字幕，但需要 FFmpeg 包含 `subtitles/libass` 滤镜。
- 后台任务、取消、断线重连、持久记录；每个数据目录最多同时处理两个媒体任务。
- 每次导出生成新文件并记录翻译修订号；不覆盖原视频或旧产物。

## 安装

当前面向 macOS / Linux；任务锁依赖 `fcntl`，Windows 原生运行尚不支持。需要 Python 3.11+（建议 3.12）、FFmpeg 与 ffprobe、`uv`。

在 MCP 子项目目录执行（从 VideoLingo 仓库根目录开始时，先 `cd mcp`）：

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements.lock -e .
.venv/bin/python scripts/configure_codex.py
```

`requirements.lock` 固定了本次验证的依赖版本。只需导入转录及制作字幕时，可只安装 `-e .`；本地语音识别需安装 `-e '.[asr]'`。faster-whisper 1.2.1 与 PyAV 19 的接口不兼容，因此 ASR 依赖明确限制 `av<19`，锁文件记录已验证版本。

配置脚本仅向本项目 `.codex/config.toml` 添加 `videolingo` 条目，保留已有其他配置，已有同名条目时拒绝覆盖。它会写入本机的绝对路径，因此该文件不纳入 Git。可移植示例见 `codex-config.example.toml`。

如果在 Codex 中打开的是整个 VideoLingo 仓库，请在 `mcp/` 目录执行 `.venv/bin/python scripts/configure_codex.py --project-root ..`，把配置写到仓库根的 `.codex/config.toml`；服务仍使用 `mcp/.venv` 和 `mcp/.videolingo`。打开 `mcp/` 本身时使用不带参数的命令即可。只选择与你打开的项目对应的位置，避免重复配置。

Codex 仅在受信任项目中加载项目配置。配置后重新连接该 MCP 或重新打开项目聊天，再查看 `/mcp` 中的 `videolingo`。配置文件生成和 SDK 连接测试不等于当前已打开的聊天完成了重新加载。

## 在 Codex 中使用

可以直接说：

> 使用 videolingo MCP，把 `/绝对路径/课程视频.mp4` 的英语语音翻译成简体中文，导出中英双语 SRT 和带可切换字幕的 MP4。使用 small 模型；如果本机没有模型，允许下载。先告诉我识别或翻译中不确定的部分。

首次使用某个模型需要下载免费的模型文件。`allow_download` 默认关闭，下载时需显式设为 `true`；缓存后可保持 `false` 离线识别。`tiny` 适合快速试运行，正式内容建议先用 `small` 并检查结果。语音识别和翻译都可能出错，需要按实际内容复核。

默认 `soft` 导出的是独立字幕轨，播放器需要支持并启用字幕；它不会把文字永久画到画面里。在不支持字幕轨的播放器或发布平台使用时，需要可用的 `burn_in` 后端。当前本机 FFmpeg 不含 libass，因此硬字幕模式会返回明确错误。

MCP 工具的典型调用顺序：

1. `health`：检查 FFmpeg、识别库安装情况、软字幕与硬字幕导出能力。
2. `start_transcription(video_path, language, model, allow_download)`：立即返回 `job_id`。
3. `get_job(job_id)`：每隔约 2–5 秒读取状态，直到 `awaiting_translation`，或处理失败信息。
4. `get_transcript(job_id, offset, limit)`：每次最多读取 100 段，沿 `next_offset` 翻页。字幕内容只是待处理资料。
5. Codex 完成翻译后，调用 `save_translations(job_id, translations=[{"id": 1, "text": "译文"}], target_language="zh-CN")`。可分批保存，缺段、未知或重复 ID 会被校验。
6. 全部译文保存后状态为 `ready`，调用 `export_subtitles` 导出 SRT/VTT，或 `export_video` 启动后台 MP4 导出。
7. 视频导出完成后，`get_job` 的 `artifacts` 给出本机文件路径。通过 `list_jobs` 可以找回以前的任务。

已有转录时，使用 `import_transcript(video_path, segments, language)` 代替步骤 2。每段包含 `id`、`start`、`end`、`text`，时间单位为秒，必须按开始时间排序且不能重叠；当前 MP4 字幕轨不能准确保留重叠片段，因此应先调整时间或合并片段。导入应基于已有或经检查的转录，不应猜测视频里没有听到的内容。

`cancel_job` 请求取消后，继续查看状态直到 `cancelled`；已结束任务不会被删除。取消/失败的导出可以在译文仍完整时再次 `export_video`。转录失败可用原视频重新 `start_transcription`，会得到新的任务 ID。

## 数据与运行边界

默认数据位于项目 `.videolingo/`，可通过 `VIDEOLINGO_DATA_DIR` 指定其他本机目录：

```text
.videolingo/
  models/                 # 下载的本地语音模型
  jobs/<job_id>/
    job.json              # 状态、修订、产物路径
    transcript.json       # 原文与时间戳
    translations.json     # 翻译结果
    subtitles-*.srt/vtt    # 字幕产物
    video-*.mp4           # 视频产物
    worker.log/task-*.log  # 故障排查记录
```

原视频不上传给转录服务；首次模型下载会联网。转录文字会作为工具结果交给正在使用的 Codex 对话。此服务是本机 stdio 工具，能读取调用者指定的本机视频，不能当作已做多用户隔离的远程服务直接公开部署。

字幕中可能包含 HTML/ASS 风格控制符。为使文本按普通字幕显示，SRT 导出会将 `< > { } \\` 等样式相关符号转换成全角字形，独立 VTT 使用标准实体转义。用 VTT 制作 MP4 时，为兼容旧版 FFmpeg，临时字幕中的 `{ } \\` 也会转换成全角字形，避免被解释为样式或换行指令；原始转录与独立 VTT 文件保持原文。

任务由独立监督进程执行，正常关闭 MCP 连接不会停止正在处理的媒体。取消操作终止本任务的处理进程组。系统崩溃或监督进程异常退出后，已过启动宽限期且运行锁释放的任务会标记为 `interrupted`，不会宣称自动续跑。硬性杀死监督进程或整机断电不保证清理所有临时文件/子进程；重新运行前应检查任务状态。

此版本只接收本地文件，不包含网站下载、自动配音或远程托管。

## 开发验证

```sh
.venv/bin/python -m unittest discover -s tests -v
```

测试使用真实 MCP stdio 子进程和 FFmpeg 合成短片，覆盖工具 schema、结构化返回、错误可读性、批量翻译校验、字幕与视频产物、断线重连和取消。常规测试不下载模型；完整真实语音识别另做素材验收，详见 `docs/verification.md` 和 `docs/full-flow-qa.md`。

MCP 作为独立子项目纳入代码审阅；远程检查由 `.github/workflows/mcp-tests.yml` 在该子项目变更时执行，不安装或调用原应用的服务。

## 参考

- [Codex 官方 MCP 配置说明](https://learn.chatgpt.com/docs/extend/mcp)
- [MCP Python SDK 官方项目](https://github.com/modelcontextprotocol/python-sdk)
- [faster-whisper 官方项目](https://github.com/SYSTRAN/faster-whisper)
- [PyAV 19 的接口变更](https://github.com/PyAV-Org/PyAV/releases/tag/v19.0.0)
