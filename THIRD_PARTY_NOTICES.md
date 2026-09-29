# Third-party notices

- The BEHAVIOR submodule is pinned to upstream v3.9.1. Its component licenses
  remain in the submodule; OmniGibson is MIT licensed and BDDL has its own
  license. Preserve those files when redistributing.
- The custom R1Pro profile derives from OmniGibson Robot Assets. The
  [upstream robot asset repository](https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets)
  declares the MIT license. The profile changes mass, joints and actuator
  configuration; the original Stanford copyright notice is retained in LICENSE.
- Scene assets, task instance datasets, NVIDIA Isaac Sim, model weights and
  decryption keys are external dependencies. Their licenses are independent
  of this repository's code license.
- Optional SAM 2 source and weights are obtained separately from
  [facebookresearch/sam2](https://github.com/facebookresearch/sam2), under its
  applicable Apache 2.0 notices. They are not part of the Git release.
- The interface bundles DejaVu Sans Bold for observation labels. Its full
  license is beside the font under `interface/behavior_interface_eval_test/tool/official_v2/assets/`.
  Optional RTAB-Map is fetched from the commit and archive hash in
  `interface/behavior_interface/rtabmap_slam/upstream.lock` under BSD-3-Clause;
  the archived task runner disables the spatial map.
