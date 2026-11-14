# runlog-client

A stand-in for `import wandb` that logs training runs to a self-hosted Runlog
server. It is one file with no dependencies beyond the Python standard library:
`wandb.init`, `wandb.log`, `wandb.watch`, `wandb.finish` and the `Image`,
`Video`, `Audio`, `Html` and `Table` wrappers keep their signatures, so training
code does not change.

Metrics do not pass through the server. `init()` asks it once for upload
addresses, and everything after that goes straight to object storage, so a
training loop cannot slow the dashboard down and there is no local agent to hang
at exit.

The server is not part of this repository.

## Install

```bash
pip install "runlog-client[media] @ git+https://github.com/MaxxP0/runlog-client.git"
```

or, in a requirements file:

```
runlog-client[media] @ git+https://github.com/MaxxP0/runlog-client.git
```

`[media]` adds what the optional parts use: numpy and Pillow for images,
imageio-ffmpeg for h264 video, psutil for machine telemetry. Leave it out for
the standard library alone.

The installed module is named `wandb`. Do not install it into an environment
that also has the real `wandb` package: that one would be imported instead.
`runlog configure` warns when this is the case.

Copying `wandb.py` next to a training script works too, with no install.

## Set up a machine

```
$ runlog configure
Runlog URL [None]:
API key [None]:
Read key, optional [None]:
wrote /root/.runlog.json (mode 600)
  API key: accepted by runlog.example.com
```

It works like `aws configure`. Enter keeps the value in brackets. A key is not
echoed while it is pasted and is never printed in full. Afterwards the server is
asked once whether it takes the keys, which creates nothing.

- `runlog configure list` shows the address and keys this machine uses, and
  where each comes from.
- `RUNLOG_URL`, `RUNLOG_API_KEY` and `RUNLOG_READ_KEY` override the file.
- Without the command: `python wandb.py configure`.
- For setup scripts: `runlog --url URL --key-stdin < keyfile`.

## Use

```python
import wandb

wandb.init(project="my-project", name="baseline", config={"lr": 1e-3})
for step in range(1000):
    wandb.log({"train/loss": loss}, step=step)
wandb.finish()
```

`runlog --smoke` runs a 20-second end-to-end check. A run that is stopped with
SIGTERM or SIGHUP still uploads its last points and says that it ended.

## Reading runs back

With a read key, scripts can fetch what the dashboard shows:

```python
import wandb

wandb.list_projects()
wandb.list_runs("my-project")
wandb.history("my-project", run_id, "train/loss")
```

or `runlog --list [PROJECT]` and `runlog --show PROJECT RUN` on the command
line.
