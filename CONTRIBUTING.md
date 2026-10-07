# Contributing

Use [GitHub issues](https://github.com/BinceQu/RoboHarness/issues) for installation
problems, reproducibility findings and proposed changes. Include the repository
commit, task and instance IDs, harness and CLI version, Python/CUDA versions,
model identifier, and the command you ran. Attach a minimal relevant log excerpt
and the official score or run summary when available. Remove credentials before
sharing local configuration or logs.

## Checks

After installing the interface environment, the runner and reporting checks
can run without starting a simulator:

```bash
PYTHONPATH=.:interface .venv-interface/bin/python -m unittest discover -s tests -v
```

After installing the interface and agent environments, run all packaged CPU
checks with:

```bash
./scripts/check.sh
```

`ROBOHARNESS_INTERFACE_PYTHON` and `ROBOHARNESS_AGENT_PYTHON` select existing
interpreters. Some native-CLI checks are opt-in; skipped checks are not passing
checks. Simulator or model changes additionally need appropriate GPU rollouts.

## Preserving the experiment

Keep reference prompts and scores unchanged. Explain archive discrepancies in
the provenance documentation instead of replacing their original bytes. Changes
to budgets, tool schemas, Skill context, model setup or initial state can affect
the experiment and need explicit evidence and a separate run directory.

Evaluate complete task means using `scripts/report_validation.py --require-match`.
Retain all five official results, including unsuccessful cases. A passing unit
test, an incomplete task or a matching subset is not evidence of a reproduced
task mean. State the scope and limits of validation in pull requests.

Use repository or session configuration for local paths, listeners and model
access. Keep datasets, weights, credentials, caches and full local trajectories
outside the tracked release. The existing `.gitignore` excludes their standard
locations.
