# Subprocess

Run shell commands through a stdlib-compatible API.

`pulsing.subprocess` mirrors Python's `subprocess` module and adds an optional
Pulsing-backed execution path when you want to attach resource requirements.

## Run

```bash
python examples/python/subprocess_example.py
python examples/python/subprocess_example.py --resources
USE_POLSING_SUBPROCESS=1 python examples/python/subprocess_example.py --resources
```

## Backend Selection

- Without `resources`, calls go directly to Python's native `subprocess`.
- With `resources=...` but without `USE_POLSING_SUBPROCESS=1`, the example still uses the native backend.
- Only when both `resources` is provided and `USE_POLSING_SUBPROCESS=1` is set does execution switch to the Pulsing backend.

## Code

```python
import pulsing.subprocess as subprocess

PIPE = subprocess.PIPE
extra = {"resources": {"num_cpus": 2}}

result = subprocess.run(
    ["echo", "hello from run()"],
    capture_output=True,
    text=True,
    check=True,
    **extra,
)

proc = subprocess.Popen(["cat"], stdin=PIPE, stdout=PIPE, text=True, **extra)
stdout, _ = proc.communicate(input="pipe test")
```

The full example also covers:

- `check_output()` for one-shot command capture
- `Popen(...).communicate()` with stdin/stdout/stderr
- `TimeoutExpired` handling
- Multi-turn shell sessions through a persistent `/bin/sh`

## Key Points

- Import it as `import pulsing.subprocess as subprocess` to keep the same calling style as the stdlib.
- The resource-backed path is opt-in, so existing `subprocess`-style code can migrate incrementally.
- In resource-backed mode, Pulsing is initialized lazily by the module itself.
