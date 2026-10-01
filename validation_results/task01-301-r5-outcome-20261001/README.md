# task01/301 r5 outcome analysis — 2026-10-01

This is diagnostic evidence from an invalid-context r5 run. It does not count
as a strict reproduction result or establish that the queued corrections work.

The official evaluator returned Q = 2/3 after 7,071 steps, against the archived
Q = 1.0. The runner recorded model_done. The completed native agent response
reported no agent error. This case ended by model decision, not wall timeout.

The recorder preserves the sequence around the second orange can:

- Turn 76, img_0057: the can is still held above the ashcan.
- Turn 77, img_0058: the right gripper opens successfully in three action steps.
- Turn 78, img_0059: the arm reaches its stowed target; the orange can is visible
  on the floor outside the ashcan. The native response at line 576 describes
  this as a placement miss. The final agent summary loosely calls it a grasp
  miss; the contemporaneous image and response give the more precise location
  of the failure.
- Turn 178: a fresh tracker binding on img_0138 reports XY separation
  0.223564425 m between the model-labelled interior and outside_can points.
  Both are observed, but the tracker explicitly marks identity_verified=false.
- The archived prompt, lines 54–74 and 180–187, instructs the agent to ignore
  missed cans and end when all remaining outside cans are within 0.5 m.
  The final model response follows that stopping rule.

A check of the original successful task01/301 transcript found three gripper
releases taking 2, 2 and 3 steps. Its arm-stow responses also report the same
feedback-governed execution mode as r5. Short opening duration alone therefore
is not evidence of a changed controller. This comparison does not establish
identical contact dynamics, target selection or complete interface equivalence.

The visible placement failure explains the missing can, and the stop rule
explains submission below full credit. Neither fact isolates why this run
missed the placement. In particular, this run had known native-context
mismatches; a single outcome cannot attribute causality to one context change.
The pinned corrected run still needs fresh official score validation.

No prompt, stop rule, tool timing or reference score is changed by this audit.
The source hashes, native line numbers, selected recorder turns and image
hashes are in audit.json. Large private run recordings are not copied into
this evidence directory.
