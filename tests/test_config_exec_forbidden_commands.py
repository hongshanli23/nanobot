from nanobot.config.schema import Config


def test_exec_forbidden_commands_accepts_camel_case_config() -> None:
    cfg = Config.model_validate(
        {
            "tools": {
                "exec": {
                    "forbiddenCommands": [r"\\bgit\\s+push\\b", r"\\bbrew\\s+upgrade\\b"],
                }
            }
        }
    )

    assert cfg.tools.exec.forbidden_commands == [
        r"\\bgit\\s+push\\b",
        r"\\bbrew\\s+upgrade\\b",
    ]
