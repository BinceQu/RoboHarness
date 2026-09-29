# Claude Code harness

The archived experiments use this harness with Qwen3.8-Flash-Next-FP8.
Launch it through the repository runner:

```bash
./run.sh --task task01 --gpu 0 --harness claude_code
```

The launcher selects the embodied plugin, binds a single loopback interface,
loads the baseline and task skill lifecycle, and records model-visible tool
calls. Its MCP server exposes the active interface catalog, camera images,
and skill activation. Robot actions are serialized. Shell, direct HTTP,
simulator management and non-robot tools are excluded from the agent profile.

`scripts/bootstrap` creates a local Python environment; the repository setup
pins MCP 2.1.1. `EMBODIED_CLAUDE_PYTHON` and `CLAUDE_BIN` select existing
executables. Sources are imported from this checkout, with no shared /tmp
source copy. Each run uses a private Claude configuration and trajectory path.

The paper route sets `EMBODIED_ANTHROPIC_BASE_URL` to an Anthropic-compatible
origin (without /v1). The optional Chat Completions bridge remains available
through `scripts/qwen-anthropic-bridge` for other deployments.

See [installation](../../docs/setup.md) and [provenance](../../docs/provenance.md).
