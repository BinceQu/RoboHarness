# This session's history

One numbered folder is created before each tool operation and holds only what
you can use: `input.json`, `result.json`, the images the call returned (for
example the red gripper visualization) and, for calls that selected pixels, a
`click_<id>_crosshair.png` with the crosshair drawn on the source image. Bulk
evidence (metric depth, joint commands, full output, archived observations) is
kept by the harness outside this workspace.

An interrupted call may have an input but no result; this does not prove the
robot stayed still. Consult current perception.

This template has no previous-session history. New sessions receive their
own copies of tools, skills, configuration, perception and this directory.
