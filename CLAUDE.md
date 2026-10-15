@AGENTS.md

## Claude Code specifics

- Before saying a change is done, run `mise run lint` and `mise run test`.
- Ask before anything that spends money or reaches out: `--reward llm`,
  `hyset reward-probe`, downloading ToolBench, pushing to a remote.
- Long training runs (`hyset train` with the sentence-transformers encoder) take minutes
  on this CPU-only machine; run them in the background and keep working.
- Scratch experiments belong in the session scratchpad, not in `src/` or `tests/`.
