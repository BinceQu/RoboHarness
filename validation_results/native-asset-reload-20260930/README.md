# Native asset reload regression

Control: an unchanged robot texture mtime event reproduced SIGABRT in carb.assets.
The fixed evaluator survived 20 events, a 90-second observation window and an
explicit official scene reset, with valid camera frames before and after.
Native watches fell from 235 to one device watcher. No model was run and this
is not a scored reproduction. See audit.json, fixed-regression.json and
control-fault-stack.txt for evidence and source/log hashes.

The cause of the separate 17:16:47 external event remains unknown.
