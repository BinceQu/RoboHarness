"""Keep generic controller flags out of public Python tool results."""


def without_ok(value):
    """Remove `ok` recursively after internal control decisions are complete."""
    if isinstance(value, dict):
        return {key: without_ok(item) for key, item in value.items() if key != "ok"}
    if isinstance(value, (list, tuple)):
        return [without_ok(item) for item in value]
    return value
