# Conversation Memory

本目录用于存储我们对话过程中值得长期保留的「对话记忆」(conversation memory),
区别于代码本身已经记录的信息(README、源码、git history、CLAUDE.md 等)。

## 存放约定

- 每条记忆一个独立 Markdown 文件,文件名用 kebab-case 简短描述内容,如
  `tirx-vs-tir.md`、`tirx-test-workflow.md`。
- 每个文件建议包含以下结构:

  ```markdown
  # <标题>

  - **主题**:一句话概括
  - **时间**:yyyy-mm-dd
  - **来源**:相关文件 / 命令 / 链接

  <正文:背景、结论、注意事项等>
  ```

- 一条 `INDEX.md` 作为索引,按时间倒序列出所有记忆条目,方便检索。

## 不放什么

- 代码本身已记录的结构、约定、历史(见 `AGENTS.md`、各模块 README、git log)。
- 只对当前对话有意义、无需长期保留的临时信息。
