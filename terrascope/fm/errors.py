class RunnerUnavailable(RuntimeError):
    """The model cannot run this request (weights / fine-tuned heads missing, or a required sensor absent).
    The message is user-facing; the API maps it to HTTP 409."""
