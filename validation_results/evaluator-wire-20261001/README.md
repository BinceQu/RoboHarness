# Fresh evaluator and interface protocol check

The independently installed evaluator and interface environments completed
an actual loopback TCP exchange using the release's communication code.
The evaluator used websockets 17.1, while the interface used 16.1.1.
All **44,065,560 bytes of observation arrays matched**, and all six returned
27-element action arrays matched the expected values. The
[audit](audit.json) records per-array hashes, shapes, dtypes, metadata,
versions, source hashes and process-preservation checks.

The check used upstream `WebsocketClientPolicy`, the release's main-thread
WebSocket client, and the actual `serve_policy` function from checkout
`c4763eb0947adf7f0186834b476a0ec375ee1d1e`. The server's runtime was a
non-actuating stub: it checked synthetic observations and returned fixed
test actions. No simulator or robot was connected.

The exchange covered:

- HTTP health checks, two WebSocket handshakes and the official metadata.
- Six observations with a 720×720 head camera and two 480×480 wrist cameras,
  alternating RGB/RGBA, float32 depth including non-finite sentinels, and
  65-value proprioception serialized from Torch tensors.
- Exact array shape, dtype and byte comparisons at the interface, and six
  CPU float32 actions with the custom profile's 27 entries at the evaluator.
- Two reset messages and a second client connection after closing the first.

Both processes kept CUDA uninitialized; OmniGibson's application and simulator
remained unset. The temporary listener used **15079**, and it was released
after the check. All r5 controller identities and recorded process roles
remained alive. The r6 source, session configuration and waiting state were
unchanged at the 21:29 Asia/Shanghai observation.

The first private probe incorrectly assumed 25 action entries and stopped
at the metadata assertion. The final probe reads the packaged custom robot
contract, which specifies 27. No release runtime code changed.

This establishes protocol interoperability for these installed versions and
payloads. It does not validate GPU simulation, physical actions, Q-scores or
complete task means. The fresh evaluator remains unselected by r6, and the
other [installation differences and dependency conflicts](../evaluator-clean-install-20261001/README.md)
remain recorded. All fifteen corrected cases and three task-mean comparisons
are still required.
