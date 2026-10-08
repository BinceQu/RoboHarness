# Case prompts

The paper specifies a human-written, task-specific prompt shared across all
instances of that task. It includes the procedure for accomplishing the task
and the behavioral boundaries the model must respect.

The selected archived transcripts contain multiple recovered prompt versions
within task00, task02, task05 and task06, as shown below. These records therefore
do not establish that those archived instances used an identical prompt.
The release preserves their recorded versions and mappings for archive replay.

Every link below points to the exact user prompt bytes in the selected
original Claude transcript, including its final newline. The runtime substitutes
only its local HTTP port. The manifest records transcript, prompt and result hashes.

| Task | 301 | 304 | 306 | 308 | 310 |
| --- | --- | --- | --- | --- | --- |
| [task00](../tasks/task00.json) | [recovered](../prompt/task00/recovered_206fb2452f46.txt) | [v24](../prompt/task00/v24_ecaa0bd65dcf.txt) | [v24](../prompt/task00/v24_ecaa0bd65dcf.txt) | [v24](../prompt/task00/v24_ecaa0bd65dcf.txt) | [v24](../prompt/task00/v24_ecaa0bd65dcf.txt) |
| [task01](../tasks/task01.json) | [v4](../prompt/task01/v4_39be138fa40a.txt) | [v4](../prompt/task01/v4_39be138fa40a.txt) | [v4](../prompt/task01/v4_39be138fa40a.txt) | [v4](../prompt/task01/v4_39be138fa40a.txt) | [v4](../prompt/task01/v4_39be138fa40a.txt) |
| [task02](../tasks/task02.json) | [v16](../prompt/task02/v16_fc8e1be396ee.txt) | [v17](../prompt/task02/v17_6ee06b7a10da.txt) | [v18](../prompt/task02/v18_85ab4a115842.txt) | [v18](../prompt/task02/v18_85ab4a115842.txt) | [v18](../prompt/task02/v18_85ab4a115842.txt) |
| [task03](../tasks/task03.json) | [v4](../prompt/task03/v4_11b5ae18163a.txt) | [v4](../prompt/task03/v4_11b5ae18163a.txt) | [v4](../prompt/task03/v4_11b5ae18163a.txt) | [v4](../prompt/task03/v4_11b5ae18163a.txt) | [v4](../prompt/task03/v4_11b5ae18163a.txt) |
| [task05](../tasks/task05.json) | [v7](../prompt/task05/v7_25db77e02e1e.txt) | [v7](../prompt/task05/v7_25db77e02e1e.txt) | [v7](../prompt/task05/v7_25db77e02e1e.txt) | [v8](../prompt/task05/v8_cb698b1105f7.txt) | [v8](../prompt/task05/v8_cb698b1105f7.txt) |
| [task06](../tasks/task06.json) | [v7](../prompt/task06/v7_625983041765.txt) | [v8](../prompt/task06/v8_b5ec2bb4b4ef.txt) | [v8](../prompt/task06/v8_b5ec2bb4b4ef.txt) | [v9](../prompt/task06/v9_6df0ac887d1e.txt) | [v9](../prompt/task06/v9_6df0ac887d1e.txt) |
| [task07](../tasks/task07.json) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) | [v18](../prompt/task07/v18_420cc8ac0d5d.txt) |
| [task08](../tasks/task08.json) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) | [v25](../prompt/task08/v25_62c12d99b1ce.txt) |
| [task09](../tasks/task09.json) | [v3](../prompt/task09/v3_bde4c7512f25.txt) | [v3](../prompt/task09/v3_bde4c7512f25.txt) | [v3](../prompt/task09/v3_bde4c7512f25.txt) | [v3](../prompt/task09/v3_bde4c7512f25.txt) | [v3](../prompt/task09/v3_bde4c7512f25.txt) |

Version labels come from matching filenames; the content hash identifies the
exact text. `recovered` denotes text with no matching versioned filename.
See [archive discrepancies](provenance.md) for cases where the session text
differs from the summary’s version label.
