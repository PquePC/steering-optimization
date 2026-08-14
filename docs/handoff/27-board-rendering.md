# 27 — The board redraws in place on a terminal, and stops flooding the log

**Status: BUILD NOW.** Branch `pareto`. Cosmetic in the sense that no measurement changes, and
not cosmetic in the sense that it already cost the operator the CAL dose map on a live run.

## The symptom

`monitor.RunStatus.render` ends in `print(txt, flush=True)`. Every update appends a **complete new
copy** of the board. On a 1h20m run that is hundreds of copies: the terminal scrollback is
unusable, and the run log is mostly duplicated boards. The operator lost Phase 0's dose map to it
and had to `grep -A 60 "dose map"` the log to get it back.

## The part that is easy to get wrong

**The operator does not watch a terminal.** The documented workflow is

```
nohup python -m m2.run --concepts Garlic --no-stop-pod > /workspace/m2_garlic.out 2>&1 &
tail -f /workspace/m2_garlic.out
```

so stdout is a **file**. `isatty()` is False, there is no cursor to move, and an ANSI redraw does
nothing for the case that actually hurts. Two behaviours are needed, not one.

| stdout | behaviour |
|---|---|
| **TTY** | redraw in place — move the cursor up, clear, rewrite |
| **not a TTY** | print the board only on **phase transitions**, plus a one-line heartbeat between them |

**Never emit an ANSI escape to a non-TTY.** `docs/RUNBOOK.md` and every diagnosis so far depend on
`grep` over that log; escape sequences would corrupt it silently and the damage would only surface
when someone needed the file most.

## The trap: `tee_stdout`

`monitor.tee_stdout` wraps stdout to write the terminal *and* a log file at once, and its wrapper
deliberately forwards `isatty` to the real stream (bug 16). So with `--log` **on a terminal**, a
single `isatty()` check returns True and the escapes go into the file as well. One check is not
enough.

Resolve it explicitly and say which you chose:

- have the board write its in-place frames to the real terminal stream, bypassing the tee, and
  send the file only phase-transition text; **or**
- have the tee strip ANSI on the file side.

The first is cleaner; the second is fewer moving parts. Either is fine, neither is implicit.

## Requirements

- The heartbeat carries what a `tail -f` reader needs and nothing else: phase, units done/total,
  elapsed, ETA. One line.
- **Phase transitions always print in full, in both modes.** They are the run's landmarks and the
  thing `grep` looks for.
- Anything that is not the board — `[INFO]` lines, dose map, gate output, tables — is unchanged.
  This task touches the board's own rendering and nothing else.
- The board must still never take the run down. `driver._Board` already wraps every call for that
  reason; the same rule applies to whatever replaces `render`.
- Degrade to plain appending if the terminal size cannot be determined, rather than emitting a
  guess at cursor movements.

## Acceptance

- A test that trips it: render twice with stdout redirected to a `StringIO` (not a TTY) and assert
  the captured text contains **no** `\x1b[` sequence.
- A test that a phase transition prints in full in both modes.
- Run the offline suite with output redirected to a file and confirm the file greps cleanly.
- Say in the commit message which `tee_stdout` resolution you took and why.
