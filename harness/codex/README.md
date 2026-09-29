# Codex harness

This is the alternate Codex implementation of the embodied MCP adapter.
The supplied historical scores were obtained with the Claude Code harness.

```bash
export OPENAI_API_KEY=...  # supply in your environment
./run.sh --task task01 --gpu 0 --harness codex \
  --model YOUR_MODEL --model-url https://your-responses-endpoint/v1
```

The launcher renders `profiles/embodied.config.toml` into a private CODEX_HOME,
binds the active robot interface, installs the local plugin cache, and applies
the original hook and tool restrictions. It imports the MCP source from this
checkout. Use a Codex CLI with plugin, hook, profile and direct MCP support;
CLI feature support can differ between releases. The launcher uses
`codex plugin marketplace add` and `codex plugin add`, with a dedicated
`roboharness` marketplace, and verifies the installed source and version.

No global authentication or relay files are copied. `CODEX_BIN` and
`EMBODIED_CODEX_PYTHON` can select existing executables. The repository runner
sets both. See [installation](../../docs/setup.md).
