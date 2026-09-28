
## 2026-09-28 stale worktree index.lock, 3 times in one day -- PI[claude]
`/workspace/2026/lite/superkv/.git/worktrees/concept-steer/index.lock` was left behind (0 bytes, no process holding it) at 13:41, 15:13 and 18:54. Retry loops never succeed against it. Check its age and `pgrep -af "git "` first; if it is older than a minute with no git process, delete it. Suspect: a status-bar `git diff` poller killed mid-write.
