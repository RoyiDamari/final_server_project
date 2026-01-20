import asyncio
from contextlib import suppress
from typing import Tuple, Any
import sys, json


def build_train_worker_cmd(
    csv_path: str,
    features: list[str],
    label: str,
    model_type: str,
    params: dict[str, Any],
    tmp_out: str,
) -> list[str]:
    """Return argv for the train_worker module (no shell)."""
    return [
        sys.executable, "-m", "app.workers.train_worker",
        "--csv", csv_path,
        "--features", json.dumps(list(features), ensure_ascii=False),
        "--label", label,
        "--model-type", model_type,
        "--params", json.dumps(params, ensure_ascii=False),
        "--tmp", tmp_out,
    ]


async def run_training_subprocess(cmd: list[str]) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out_b, err_b = await proc.communicate()  
    except asyncio.CancelledError:
        with suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        raise
    return proc.returncode, (out_b.decode() if out_b else ""), (err_b.decode() if err_b else "")
