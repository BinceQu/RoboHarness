# Model service provenance

The current endpoint reports the same vLLM version named in a recovered
September 5 deployment report: `0.28.1rc1.dev202+gffc445f`. Its model ID,
checkpoint path and context limit also agree with the earlier deployment
snapshot. This narrows the configuration uncertainty; it does not establish
identical weights, sampling or server arguments for the paper rollouts.

The [audit](audit.json) records original file hashes, selected deployment
metadata, all 15 selected archived transcript hashes and current read-only
`/version` and `/v1/models` observations. No inference request, server change
or benchmark restart was performed for this audit.

| Time (Asia/Shanghai) | Evidence | What it establishes |
| --- | --- | --- |
| September 4, 23:17 | Original `server_snapshot.json` | The endpoint previously reported SGLang `0.0.0.dev1+g78c5024e9`, the same model name and checkpoint path. |
| September 5, 03:17 | Saved `vllm_tp2_20260905_release_21/results.json` and September 5 deployment report | The saved visual replay records 21/21 passes; the report identifies the replacement as vLLM `0.28.1rc1.dev202+gffc445f`. |
| September 15–19 | First responses in all 15 hash-verified selected transcripts | The paper cases postdate that saved replay and name `Qwen3.8-Flash-Next-FP8`. The transcripts do not record the engine version. |
| October 3 | Current endpoint metadata in the audit | `/version` matches the September 5 reported version. `/v1/models` names the same model and path, with `max_model_len=262144`. |

The 21-call artifact is a saved visual-grounding check of seven images over
three repetitions. Its result flags and all recorded bridge assertions were
checked, and the source file was hashed. It was not rerun. It is not a
BEHAVIOR task result, does not use mean Q, and cannot replace the pending
five-instance GPU evaluations. The older coordinate-adapter discussion in
the historical report is superseded and is not used to configure this release.

## Recovered launch settings

[reference-launcher.sh](reference-launcher.sh) is a byte-for-byte copy of the
original `deploy/qwen38_vllm.sh`, retained as historical evidence. It is not
called by RoboHarness. It contains the original host's executable, model and
compiler paths and is not a portable installer. Its defaults include:

| Setting | Recorded script default |
| --- | --- |
| Model | `Qwen3.8-Flash-Next-FP8` |
| Checkpoint path | `/home/bince/models/models--Qwen--Qwen3.8-Flash-Next-FP8` |
| Tensor parallel size | 2 |
| Context limit | 262144 |
| Maximum sequences | 4 |
| GPU memory utilization | 0.88 |
| Vision attention backend | `TORCH_SDPA` |
| Prefix caching | enabled |
| Reasoning / tool parsers | `qwen3` / `qwen3_coder` |
| PLE CPU offload | enabled |
| FlashInfer sampler | disabled |

Environment variables can override several of these defaults. A surviving
script does not prove the command or environment actually used for every
rollout, or the parameters of today's service. Its custom vLLM installation
and checkpoint are external dependencies, not supplied by this repository.

The historical and current weight/tokenizer hashes, exact per-episode server
parameters and complete historical inference request bytes remain unverified.
The recovered evidence supports the model-service correspondence investigation;
it does not establish that a service change caused any Q-score difference.
The ongoing reproduction status remains [unverified](../../docs/validation.md).
