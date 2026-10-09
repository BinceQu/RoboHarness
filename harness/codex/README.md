# Codex harness

This is the alternate Codex implementation of the embodied MCP adapter.
The supplied historical scores were obtained with the Claude Code harness.

Start a model service with the Responses API and configure your local data and
Python paths first; see [Codex model service](../../docs/setup.md#codex-model-service).
Keep that service running while the task executes.

```bash
export OPENAI_API_KEY='YOUR_RESPONSES_SERVICE_KEY'
./scripts/reproduce_task.sh task01 --gpu 0 --harness codex \
  --model YOUR_SERVED_MODEL_ID --model-url https://YOUR_MODEL_HOST/v1
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
