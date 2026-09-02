"""Small provider-neutral chat formatting primitives."""


def bash_console_block(command: str, output: str = "") -> str:
    """Render a shell command and its output as one fenced Bash block."""
    body = "$ " + command
    rendered_output = output.rstrip("\n")
    if rendered_output:
        body += "\n" + rendered_output
    body = body.replace("```", "``\u200b`")
    return "```bash\n" + body + "\n```"
