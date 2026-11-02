# Cortex: next tools plan

For our meeting on 2026-11-02. Nothing here is built yet.

**The rule from Niranjan:** context is the most important part. Keep response quality and compute at the top, and don't compromise on them. Every idea below has to pass that test.

## Short version

- Don't add tools first. Make room first, because the current list already takes 3,161 tokens on every message.
- Then add three tools, chosen from what actually fails today: `server`, `mac_status`, `storage`.
- Skip an `adb` tool. It looked obvious, but `adb` has never failed.
- Never change the tool list from message to message. That turns 0.13 s into 6 to 8 s.

## What we measured

All of this is from the real model (`gemma4-agent:12b-mlx`) and real chat history.

| What | Number |
|---|---|
| Cost of every message before anything is said | 5,218 tokens, 16% of the 32,768 window |
| Instructions | 2,057 tokens |
| The 22 tools | 3,161 tokens |
| Reading that with an unchanged tool list | 0.13 s |
| Reading that after the tool list changes | 6 to 8 s |
| Commands that failed | 135 of 420 (32%) |
| Commands aimed at the server (`docker`, `ssh`) | 143, and 69 of them failed |
| Those `docker` commands failing because Docker isn't running on the Mac | 38 |

Each failed command is another trip through the model, so failures cost both context and time.

The three longest tools take a third of all tool tokens: `watch_process` 406, `schedule_task` 373, `manage_workers` 283.

Seven tools have no recorded use in real chats: `watch_process`, `manage_workers`, `delegate`, `forget`, `crawl_site`, `request_credentials`, `undo_changes`. Together they cost 1,224 tokens. Some of them still matter, such as undo, so this is a reason to shorten them, not to delete them.

| Program | Runs | Failed | Average output |
|---|---|---|---|
| `docker` | 89 | 60% | 915 characters |
| `adb` | 67 | 0% | 30 characters |
| `ssh` | 53 | 30% | 1,921 characters |
| `ps` | 33 | 12% | 1,106 characters |
| `top` | 10 | 50% | 1,236 characters |
| `find` | 9 | 44% | 930 characters |
| `brew` | 5 | 80% | 527 characters |
| `pip` | 5 | 100% | 32 characters |

## Three rules for any new tool

1. **Fixed budget.** Instructions plus tools stay at or under 4,500 tokens. A new tool has to be paid for by trimming something else.
2. **Frozen list.** The same tools on every message, so the cache keeps working. This also rules out loading tools per message, and it means any MCP server (which adds all its tools) needs a cap.
3. **Earns its place.** It must replace something that fails often or dumps a lot of output, and its output must be short.

## Step 1: make room, with no new abilities

Do these one at a time and run the full real-model test after each. Savings are estimates until measured.

| Change | Estimated saving |
|---|---|
| Shorten the three longest tool descriptions | about 550 tokens |
| Merge `email_check`, `email_search` and `email_send` into `email` | about 80 |
| Merge `remember` and `forget` into `memory` | about 90 |
| Merge `browse_url`, `crawl_site` and `browser_automate` | about 140 |
| Remove `delegate` from the list (the real model picked it 0 of 4 times when asked directly, and Cortex already decides this itself) | about 190 |

Estimated result: tools fall from 3,161 to about 2,100 tokens.

Merging is also a quality fix. Today the model sometimes picks the wrong tool, as when "remove that task" went to reminders or to background workers. Fewer, clearer tools should reduce that, but we have to measure it.

## Step 2: new tools, in order

| Tool | What it does | Why |
|---|---|---|
| **`server`** | Status, containers, logs and restart on myserver, with short output. Restart still needs approval. | The biggest failure source. It would also say at once that Tailscale is off, instead of timing out for 15 s. Autonomy already reaches the agentkit `devops_*` tools, so most of the work exists. |
| **`mac_status`** | Fresh CPU, memory, disk, top apps and ports in one short call. | Replaces guessing `top` and `ps` flags. The model has invented commands like `diskfree`. |
| **`storage`** | Finds big reclaimable items (Xcode, simulators, Docker, caches) and shows space before and after. Deleting still goes through Trash and approval. | A repeated real request, and `find` fails 44% of the time. |

Estimated cost: about 300 tokens for all three. Instructions plus tools would land near 4,460 tokens, about 750 fewer than today, with more ability.

**Smaller idea:** a way to see and mute what's notifying you. One chat had about 8 messages trying to stop a notification, so it deserves a fix, but it can probably be an option on an existing tool.

## Not adding

- **`adb` tool:** 67 runs, none failed.
- **`brew` or `pip` tools:** they fail a lot but ran only 10 times. A line in the instructions is cheaper.
- **Calendar, notes, to-do:** no real demand in the chats.
- **More MCP servers:** each one adds its tools to every message.

## Context quality, beyond tools

1. **Test long chats.** Nothing yet proves that a fact from early in a long chat survives when older messages get summarised. Some chats are already at 52%. This is the biggest untested risk to quality, so do it first.
2. **Send only what the question needs.** The background data held for a machine question includes GitHub and project data (up to 2,771 characters per item). "and ram?" doesn't need that. Check what really gets sent, then trim it.
3. **Remove the disk duplicate.** The main disk appears twice in that data.

## Compute

- **Temperature, still open.** The model runs at temperature 1, which is very random. Stalls appear in about 1 of 15 live conversations but never in replays, so replays can't settle this. Run a live comparison, about 50 conversations each at 1.0 and 0.6, counting stalls and wrong answers. If 0.6 is calmer, that's a free gain in quality and compute.
- **Autonomy.** It keeps checking memory and Docker, at about 7 s a wake. Pause it, or redesign it.
- **First reply after 2 idle minutes.** It takes about 9 s because the model unloads. Keeping it loaded longer fixes that but holds about 16 GB of RAM.

## How we judge every change

Run before and after, and keep a change only if nothing gets worse:

- The full real-model conversation test (55 messages), checked against stored data
- Wrong-tool picks
- Fixed token cost
- Median and slowest reply time
- Failed command rate (32% today)

## Decisions, made 2026-11-02

1. **Budget:** cap instructions plus tools at 4,500 tokens, then one in, one out.
2. **Order:** long-chat test first, then Step 1 (trim), then `server`.
3. **Reminders and recurring tasks:** keep them separate. Clarify both descriptions first, and merge only if the model still mixes them up.
4. **Autonomy:** pause it until it's redesigned.
5. **Model:** keep the 2-minute unload.
6. **Git:** commit locally now. Change the exposed database password and push `test` after the tool work.
