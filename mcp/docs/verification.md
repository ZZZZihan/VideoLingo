# 第一版验证记录

这是初次实现阶段的历史记录。后续完整流程、修复和 PR 交付的当前结论见 [完整流程验收](full-flow-qa.md)。

验证日期：2026-10-29，Asia/Shanghai。范围为当前本机 macOS ARM64 环境与项目 Python 3.12；不等同于所有平台验证或用户人工验收。

## 最终整合检查

主 Agent 在最终代码与依赖修复后执行：

```text
.venv/bin/python -m unittest discover -s tests -v
Ran 10 tests in 7.106s
OK

uv pip check --python .venv/bin/python
Checked 44 packages
All installed packages are compatible

git diff --check
通过
```

10 项测试包括真实 MCP stdio 客户端启动、10 个工具的输入/输出 schema、结构化返回、可读错误、导入/分页/分批翻译、字幕完整性、修订产物保留、SRT/VTT、FFmpeg MP4 字幕轨导出、断线重连、取消与中断状态恢复。测试内容见 `tests/test_protocol.py` 和 `tests/test_workflow.py`。

实际取消测试使用可控处理程序及其真实子进程，取消后均退出且未发布视频。媒体实现 Agent 另做了运行中 FFmpeg 取消检查，确认未发布成品；监督进程随后补上了取消后的临时导出目录清理。

中断恢复测试构造“任务仍标为 active、启动时间已经过期、运行锁已经释放”的现场，验证标记 `interrupted` 并保留已有转录。没有声称验证了硬杀监督进程后所有孤儿进程的清理。

## 真实语音识别与导出

使用 macOS `say` 在本机生成英文测试语音，配合 FFmpeg 制作 10.42 秒纯色短片。它是合成测试素材，不是用户的正式视频。

首次下载 tiny 模型后，发现 faster-whisper 1.2.1 与 PyAV 19.0.2 的 `metadata_errors` 参数不兼容。依据 PyAV 官方变更，将 ASR 依赖限制为 `av>=11,<19`；实际解析并验证的版本为 PyAV 18.1.0，已写入锁文件。原失败任务留在本地运行记录中，未伪装为成功。

修复后，使用 `allow_download=False` 读取已缓存 tiny 模型，真实识别结果为：

> Hello everyone, this is a test of our video subtitle tool, we can turn spoken English into Chinese subtitles, thank you for watching.

随后使用 `.codex/config.toml` 中的**实际命令、目录与环境参数**启动官方 SDK stdio 客户端，经 MCP 工具读取转录、提交以下中文翻译并导出：

> 大家好，这是我们的视频字幕工具测试。我们可以把英语语音转换成中文字幕。感谢观看。

结果：

- 任务 `b3cac153686940faacf0a416887f25e8` 达到 `completed`。
- 成功导出中文 SRT、双语 VTT 和 MP4。
- `ffprobe` 确认 MP4 同时包含 H.264 视频、AAC 音频、`mov_text` 字幕轨。
- 再从成品字幕轨提取 SRT，内容与提交的中文翻译完全匹配。
- 原始测试视频 SHA-256 在处理前后相同。
- 本机取证与演示产物位于 `.scratch/smoke/verification.json` 及 `.videolingo/jobs/b3cac153686940faacf0a416887f25e8/`，均不提交到 Git。

此测试证明短片上的真实识别和工具流程可运行，不代表长视频、多说话人、嘈杂录音或各语种的识别质量已经评估。翻译由当前 Codex Agent 完成并通过工具保存，没有单独调用付费翻译 API。

## 接入与已知限制

项目 `.codex/config.toml` 已生成，路径和语法已检查，并实际用于 SDK 连接验证。当前聊天的工具列表是否已重新加载，以及 Codex 界面中的实际调用，仍须重新连接后的客户端验证；未把配置写入等同于 UI 验收。

本机 FFmpeg 8.1 没有 `subtitles/libass` 滤镜，已验证可切换字幕轨的 `soft` 模式；`burn_in` 返回明确的能力缺失信息，未声称已验证硬字幕效果。

当前没有配置 Git 远端，尚无 PR、远程 CI、合并或部署结果。本地工作在 `feature/video-subtitle-mcp` 分支。

## 协作与复核

两名子 Agent 分别负责媒体/字幕实现与 MCP 集成测试及技术复核；媒体实现 Agent 另对主 Agent 编写的任务管理和取消逻辑进行了交叉检查。主 Agent 完成服务、持久任务监督、依赖兼容修复、Codex 配置、最终整合验证及真实语音识别验收。

两名子 Agent 创建时均显式请求 `gpt-6.1-sol` / `ultra`。本机配置文件显示主配置为 `gpt-6-astra` / `ultra`；这说明配置和请求情况，不替代后端实际运行模型的独立核验。
