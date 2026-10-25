# Third-party notices

- The repository's separate `eval/` and `BEHAVIOR/` directories contain the pinned BEHAVIOR
  checkout and its licenses. This robot workspace does not import it.
- The custom R1Pro profile derives from OmniGibson Robot Assets. The
  [upstream robot asset repository](https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets)
  declares the MIT license. The profile changes mass, joints and actuator
  configuration; the original Stanford copyright notice is retained in LICENSE.
- Scene assets, task instance datasets, NVIDIA Isaac Sim and decryption keys
  belong to the evaluator deployment. They are not robot workspace libraries;
  their licenses are independent of this repository's code license.
- SAM 2 source in `.dependencies/sam2` and the checkpoint in
  `robot_config/assets/sam2.1_hiera_small.pt` originate from
  [facebookresearch/sam2](https://github.com/facebookresearch/sam2), under its
  Apache 2.0 license and accompanying notices, retained with the source.
- The local cuRobo wheel in `.dependencies/wheels` retains its original
  NVIDIA license in `nvidia_curobo-0.0.0.dist-info/licenses/LICENSE` and the
  notices bundled with its robot resources. `requirements.lock` lists the
  other installed third-party distributions; each retains its package notices.
- DejaVu Sans and DejaVu Sans Bold are included in `robot_config/assets/fonts`
  with `LICENSE.txt`. Observation labels also use the font and license in
  `tools/util/geometry/assets`.
- The scene-free joint illustrations use the included `robot_config/model.urdf`
  and `robot_config/meshes`. Colours and axis arrows are explanatory
  annotations. The original robot assets retain their applicable notices.
