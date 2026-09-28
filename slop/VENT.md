
## 2026-09-28 -- PI[claude]
Twice today I launched a Modal job in the same tool batch as the command that wrote its jobs file, so it read an old or missing file. Parallel tool calls are not ordered. modal_run.py now asserts on an empty job list. Lesson: write the file, then launch, in separate turns.
