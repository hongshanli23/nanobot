## nanobot Entry Points and Call Flow

This diagram focuses on direct local usage from terminal first, then notes the
other places that also call `AgentLoop.process_direct()`.

### 1. Top-level startup

```text
User shell
  -> runs `nanobot ...`
     or `python -m nanobot ...`

python -m nanobot
  -> nanobot/__main__.py
     -> from nanobot.cli.commands import app
     -> app()

nanobot executable
  -> [project.scripts]
     -> nanobot.cli.commands:app

Typer app
  -> parses CLI args
  -> dispatches to a command function in nanobot/cli/commands.py
```

### 2. Terminal entry point: `nanobot agent`

```text
CLI command: `nanobot agent`
  -> nanobot.cli.commands.agent(...)
     -> _load_runtime_config(...)
     -> sync_workspace_templates(...)
     -> MessageBus()
     -> _make_provider(config)
     -> CronService(...)
     -> AgentLoop(...)
```

From there, terminal usage splits into two modes.

### 3. Single-message terminal mode

This path is used when you run something like:

```text
nanobot agent --message "summarize this repo"
```

Call stack:

```text
nanobot agent --message ...
  -> cli.commands.agent(...)
  -> asyncio.run(run_once())
  -> run_once()
  -> agent_loop.process_direct(message, session_id, on_progress=_cli_progress)
  -> AgentLoop.process_direct(...)
     -> _connect_mcp()
     -> construct InboundMessage(channel="cli", chat_id="direct", ...)
     -> _process_message(msg, session_key=...)
     -> return response.content
  -> _print_agent_response(...)
```

Detailed internal flow:

```text
process_direct()
  -> _connect_mcp()
  -> InboundMessage(...)
  -> _process_message(...)
     -> sessions.get_or_create(session_key)
     -> _set_tool_context(...)
     -> session.get_history(...)
        -> compile_history(...)
     -> context.build_messages(...)
        -> build_system_prompt()
        -> load bootstrap files
        -> load memory context
        -> append prior history
        -> append current user message
     -> _run_agent_loop(initial_messages)
        -> provider.chat(...)
        -> if tool calls:
             -> context.add_assistant_message(...)
             -> tools.execute(...)
             -> context.add_tool_result(...)
             -> provider.chat(...) again
           else:
             -> context.add_assistant_message(...)
             -> final answer ready
     -> _save_turn(session, all_msgs, ...)
     -> sessions.save(session)
     -> OutboundMessage(...)
  -> return final response text
```

### 4. Interactive terminal mode

This path is used when you run:

```text
nanobot agent
```

with no `--message`.

This mode does **not** call `process_direct()`. It uses the full queue-driven
path, even though everything is still local in the terminal.

```text
nanobot agent
  -> cli.commands.agent(...)
  -> asyncio.run(run_interactive())
  -> run_interactive()
     -> asyncio.create_task(agent_loop.run())
     -> read terminal input in a loop
     -> bus.publish_inbound(InboundMessage(...))

AgentLoop.run()
  -> wait for bus.consume_inbound()
  -> create task: _dispatch(msg)

_dispatch(msg)
  -> _process_message(msg)
  -> bus.publish_outbound(response)

run_interactive() outbound consumer
  -> bus.consume_outbound()
  -> print progress lines and final response to terminal
```

So for terminal interaction there are really two execution styles:

```text
single message (`--message`)
  -> direct call path
  -> process_direct()

interactive chat (no `--message`)
  -> local bus path
  -> run() / _dispatch() / _process_message()
```

### 5. The core agent loop

This is the important task-completion loop regardless of how the request
entered the system.

```text
AgentLoop._run_agent_loop(messages)
  -> while iteration < max_iterations:
       -> provider.chat(messages, tools, model, ...)

       -> if response.has_tool_calls:
            -> optionally emit progress/tool hints
            -> context.add_assistant_message(..., tool_calls=[...])

            -> for each tool call:
                 -> ToolRegistry.execute(tool_name, arguments)
                 -> context.add_tool_result(tool_call_id, tool_name, result)

            -> loop continues with updated messages

       -> else:
            -> clean assistant content
            -> context.add_assistant_message(final_text)
            -> break

  -> return (final_content, tools_used, all_messages)
```

Conceptually:

```text
build context
  -> call model
  -> if model wants tools:
       execute tools
       feed tool results back
       call model again
  -> repeat until final answer
```

### 6. Persistence after each completed turn

After `_run_agent_loop()` finishes, nanobot persists the new turn.

```text
_process_message(...)
  -> _save_turn(session, all_msgs, skip=...)
     -> persist structured transcript items:
          - user
          - assistant
          - tool_call
          - tool_result
          - summary
     -> persist full raw tool output to session_artifacts/ when needed

  -> sessions.save(session)
     -> write JSONL session file under workspace/sessions/
```

Working memory and long-term memory are separate:

```text
short-term working transcript
  -> workspace/sessions/*.jsonl

raw full tool outputs
  -> workspace/session_artifacts/

long-term summarized memory
  -> workspace/memory/MEMORY.md
  -> workspace/memory/HISTORY.md
```

### 7. Important methods and roles

```text
nanobot/__main__.py
  top-level module entry for `python -m nanobot`

nanobot/cli/commands.py: app
  top-level Typer CLI application

cli.commands.agent(...)
  terminal command entry point

AgentLoop.process_direct(...)
  single-message direct entry into the agent

AgentLoop.run()
  queue-driven main loop

AgentLoop._dispatch(...)
  one inbound message -> one processing task

AgentLoop._process_message(...)
  session/history/context orchestration around one user request

AgentLoop._run_agent_loop(...)
  actual model/tool/model/tool/final-answer loop

AgentLoop._save_turn(...)
  persists the completed turn back to session storage
```

### 8. Who calls `process_direct()`

There are three important call sites.

```text
1. Terminal single-message mode
   cli.commands.agent(...)
     -> run_once()
     -> agent_loop.process_direct(...)

2. Gateway cron jobs
   cli.commands.gateway(...)
     -> on_cron_job(...)
     -> agent.process_direct(...)

3. Gateway heartbeat execution
   cli.commands.gateway(...)
     -> on_heartbeat_execute(...)
     -> agent.process_direct(...)
```

So `process_direct()` is not the only path through nanobot, but it is the most
direct path into the agent core for one-shot local requests.

### 9. End-to-end terminal example

```text
User
  -> nanobot agent --message "read README and summarize architecture"

Typer
  -> cli.commands.agent(...)
  -> build config/provider/AgentLoop
  -> agent_loop.process_direct(...)

process_direct()
  -> _process_message(...)
     -> load session history
     -> build prompt
     -> _run_agent_loop(...)
        -> provider.chat(...)
        -> assistant requests read_file("README.md")
        -> execute read_file tool
        -> append tool result
        -> provider.chat(...) again
        -> assistant writes final summary
     -> _save_turn(...)
     -> save session JSONL

CLI
  -> print final response to terminal
```

### 10. Minimal mental model

```text
CLI entry
  -> construct AgentLoop
  -> build prompt from memory + history + user input
  -> run model/tool loop until final answer
  -> save transcript
  -> print response
```
