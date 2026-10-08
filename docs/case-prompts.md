# Task prompts

In the paper's BEHAVIOR evaluation, the prompt is specific to each task but
shared across instances, and is written by a human. It specifies the execution
steps and the behavioral boundaries the model must respect.

Select a task to run its configured instances and prompts:

```bash
./scripts/reproduce_task.sh task01 --gpu 0
```

Each task configuration lists its instance IDs and prompt files. The runner
loads the configured prompt for each instance and sets the interface port
automatically. The table links to those configurations and prompt files.

| Task | 301 | 304 | 306 | 308 | 310 |
| --- | --- | --- | --- | --- | --- |
| [task00](../tasks/task00.json) | [prompt](../prompt/task00/recovered_206fb2452f46.txt) | [v24](../prompt/task00/v24_ecaa0bd65dcf.txt) | [v24](../prompt/task00/v24_ecaa0bd65dcf.txt) | [v24](../prompt/task00/v24_ecaa0bd65dcf.txt) | [v24](../prompt/task00/v24_ecaa0bd65dcf.txt) |
| [task01](../tasks/task01.json) | [v4](../prompt/task01/v4_39be138fa40a.txt) | [v4](../prompt/task01/v4_39be138fa40a.txt) | [v4](../prompt/task01/v4_39be138fa40a.txt) | [v4](../prompt/task01/v4_39be138fa40a.txt) | [v4](../prompt/task01/v4_39be138fa40a.txt) |
| [task02](../tasks/task02.json) | [v16](../prompt/task02/v16_fc8e1be396ee.txt) | [v17](../prompt/task02/v17_6ee06b7a10da.txt) | [v18](../prompt/task02/v18_85ab4a115842.txt) | [v18](../prompt/task02/v18_85ab4a115842.txt) | [v18](../prompt/task02/v18_85ab4a115842.txt) |
| [task03](../tasks/task03.json) | [v4](../prompt/task03/v4_11b5ae18163a.txt) | [v4](../prompt/task03/v4_11b5ae18163a.txt) | [v4](../prompt/task03/v4_11b5ae18163a.txt) | [v4](../prompt/task03/v4_11b5ae18163a.txt) | [v4](../prompt/task03/v4_11b5ae18163a.txt) |
| [task05](../tasks/task05.json) | [v7](../prompt/task05/v7_25db77e02e1e.txt) | [v7](../prompt/task05/v7_25db77e02e1e.txt) | [v7](../prompt/task05/v7_25db77e02e1e.txt) | [v8](../prompt/task05/v8_cb698b1105f7.txt) | [v8](../prompt/task05/v8_cb698b1105f7.txt) |
| [task06](../tasks/task06.json) | [v7](../prompt/task06/v7_625983041765.txt) | [v8](../prompt/task06/v8_b5ec2bb4b4ef.txt) | [v8](../prompt/task06/v8_b5ec2bb4b4ef.txt) | [v9](../prompt/task06/v9_6df0ac887d1e.txt) | [v9](../prompt/task06/v9_6df0ac887d1e.txt) |
| [task07](../tasks/task07.json) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) |
| [task08](../tasks/task08.json) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) |
| [task09](../tasks/task09.json) | [v3](../prompt/task09/v3_bde4c7512f25.txt) | [v3](../prompt/task09/v3_bde4c7512f25.txt) | [v3](../prompt/task09/v3_bde4c7512f25.txt) | [v3](../prompt/task09/v3_bde4c7512f25.txt) | [v3](../prompt/task09/v3_bde4c7512f25.txt) |

See [evaluation records](provenance.md) for detailed experiment settings and results.
