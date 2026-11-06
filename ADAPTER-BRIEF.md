# Adapter brief: stdio ACP adapter for a coding agent

Goal: write one program that speaks the Agent Client Protocol (ACP) over stdio and drives a
coding agent through a model endpoint. The program is the adapter. It must pass the checker below
without cheating.

## 1. Transport

- The adapter is started as a child process. The client writes JSON-RPC 2.0 messages, one per
  line, to the adapter's stdin. The adapter writes JSON-RPC 2.0 messages, one per line, to stdout.
- Stdout carries protocol messages only. Logs, banners, progress text and tracebacks go to stderr.
- Every request gets exactly one response with the same `id`. Notifications have no `id`.
- Stop the adapter's whole process tree when the client closes stdin or the process is killed.

## 2. Required behaviour

1. **Handshake.** `initialize` returns a result with `protocolVersion` 1 and the adapter's
   capabilities. `session/new` with `cwd` and `mcpServers` returns a `sessionId` string.
2. **Streaming.** `session/prompt` produces `session/update` notifications with
   `sessionUpdate: "agent_message_chunk"` as the model's text arrives, before the final response.
   The final response carries `stopReason: "end_turn"` on success.
3. **Clean text.** The chunks carry only the model's words. No banners, token counts, costs,
   version lines, git notices or terminal escape codes.
4. **File edit.** When the model changes a file in the session folder, the adapter reads the
   file's real content and sends it to the model before the edit is requested. The edit is made
   on disk. The tool call is reported with `tool_call` and `tool_call_update` notifications whose
   status goes from a non-completed value to `completed`, and which carry a `diff` content item
   or a `locations` entry.
5. **Edit permission.** Before a file is added or edited, the adapter sends
   `session/request_permission`. If the client answers `reject`, the file is not sent to the model
   and is not changed, and the turn still ends with `end_turn`. If the client answers `allow`,
   the change is made.
6. **File creation.** Creating a new file in the session folder asks for permission the same way.
   Reject means no file. Allow means the file exists with the requested content.
7. **Shell command.** A command the model asks to run asks for permission first. Reject means the
   command does not run. Allow means it runs in the session folder.
8. **Paths.** The model only ever receives files inside the session folder. Files named by
   relative path (with `..`) or absolute path outside it are never read into a model request.
9. **Cancel.** `session/cancel` during a turn ends that turn with `stopReason: "cancelled"` within
   a few seconds. The next prompt in the same session works.
10. **Memory.** A second prompt in the same session carries the first turn's reply in the model
    request. The adapter keeps one process for the session; it does not restart between turns.
11. **Model and effort.** When the adapter lists a model option or an effort option in its session
    result, changing it changes the `model` and effort fields of the next model request. A change
    during a running turn is refused.
12. **Errors.** A model error (for example HTTP 401) ends the turn with a JSON-RPC error response
    that has a message. It must not end as a normal `end_turn`.
13. **Stdout.** Every stdout line is valid JSON-RPC 2.0.
14. **Reasoning.** Reasoning text is sent as `agent_thought_chunk`, never as message text. The
    answer is sent as `agent_message_chunk`. Reasoning tags and their header lines are not shown,
    even when a tag is split across chunks.
15. **Edit blocks.** SEARCH/REPLACE blocks are not message text. The text before and after a block
    still streams. Each edit is a `tool_call` with kind `edit`, and its `tool_call_update` carries a
    `diff` content item with the real path, old text and new text. A block split across chunks is
    handled the same way.
16. **Shell output.** An allowed shell command is a `tool_call` of kind `execute` with the command as
    its title. Its `tool_call_update` carries the output as text and status `completed` or `failed`.
17. **One shell permission.** Each shell command asks the client once (`session/request_permission`,
    kind `execute`). Aider's follow-up question about adding the output to the chat is not sent to the
    client: it is answered yes only when the command was allowed, and no when it was denied.
18. **Context use.** When the model's input window is known, each model request sends a
    `usage_update` with `used` (the prompt tokens Aider counted) and `size` (the input window).

## 3. How to run the checker

The checker is `checks/check_adapter.py` in the checks folder (Python 3, standard library only).

    python checks/check_adapter.py --cmd "<command that starts the adapter>" [--json out.json] [--replies NAME]

- `--cmd` is the full command line that starts the adapter, with absolute paths.
- The checker starts a localhost mock model for each run, sets an OpenAI-compatible base URL and a
  fake key in the environment, and runs every probe in a fresh temporary session folder.
- `--replies NAME` selects the reply pack: the format the scripted model uses to ask for an edit,
  a new file or a shell command. Use the pack that matches the target agent's output format.
  The pass rules are the same for every pack.
- Output: one line per probe, `PASS`, `FAIL` or `SKIP` with a reason. Exit code 1 if any probe fails.
- A full run must show every probe PASS. A SKIP is acceptable only with a stated reason.

Acceptance: `FAIL 0` and no probe skipped for a reason inside the adapter's control.

## 4. Rules for the adapter (no cheating)

- Call the model. The model endpoint comes from the environment the checker sets. Do not
  hard-code replies, answers, file contents or tool results.
- Do not special-case the probe words (the probe tokens in prompts, the marker strings, the file
  names `hello.txt`, `new.txt`, `marker.txt`, `secret_*`). Each probe must be passed by the same
  general code path a real user prompt uses.
- Do not read the checker's files, its mock, or its reply packs to decide what to do.
- Do not write the expected output, the diff, or the status values by hand without the real
  effect on disk having happened.
- Permission must be real: ask through `session/request_permission` and act on the answer.
- Do not skip a probe by returning an empty capability list to avoid it.
- Do not modify the checker, the reply packs, the mock or the test folders to make the adapter pass.
- Keep secrets out of the code and out of output. Use only the fake key the checker provides.
- Do not start processes that outlive the adapter. Do not kill processes by image name.
- No network access other than the localhost mock during checks.

## 5. Deliverable

- One adapter script and the exact command used to start it.
- The checker output for a full run, saved with `--json`.
- A short note listing any probe that was skipped and why.
