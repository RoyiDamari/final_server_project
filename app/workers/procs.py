import asyncio
import sys
import json
from contextlib import suppress
from typing import Any, Tuple
from fastapi import Request
from app.exceptions.user import UserDisconnectedException


def build_train_worker_cmd(
        csv_path: str,
        features: list[str],
        label: str,
        model_type: str,
        params: dict[str, Any],
        tmp_out: str,
) -> list[str]:
    """
    Build the argv list to launch the training worker module via Python (no shell).

    This creates a subprocess command equivalent to:
        python -m app.workers.train_worker --csv ... --features ... --label ... --model-type ... --params ... --tmp ...

    Args:
        csv_path: Path to the validated training CSV file on disk.
        features: Feature column names to train on.
        label: Target column name.
        model_type: Model family identifier (e.g., "linear", "logistic", "random_forest").
        params: Normalized model hyperparameters' dict.
        tmp_out: Temporary file path where the worker should write the trained model artifact.

    Returns:
        A list of strings representing the subprocess argv suitable for asyncio.create_subprocess_exec().
    """

    return [
        sys.executable, "-m", "app.workers.train_worker",
        "--csv", csv_path,
        "--features", json.dumps(list(features), ensure_ascii=False),
        "--label", label,
        "--model-type", model_type,
        "--params", json.dumps(params, ensure_ascii=False),
        "--tmp", tmp_out,
    ]


async def _watch_disconnect(request: Request, poll_s: float = 0.2) -> None:
    """
    Periodically poll the FastAPI request to detect client disconnect.

    If the client disconnects, this coroutine raises UserDisconnectedException.

    Args:
        request: FastAPI Request object to check connectivity state.
        poll_s: Poll interval in seconds.

    Returns:
        None.

    Raises:
        UserDisconnectedException: When the client disconnects.
    """

    while True:
        if await request.is_disconnected():
            raise UserDisconnectedException()
        await asyncio.sleep(poll_s)


async def _terminate_proc(proc: asyncio.subprocess.Process) -> None:
    """
    Best-effort terminate a subprocess, escalating to kill if needed.

    Termination sequence:
        1) proc.terminate()
        2) wait up to 3 seconds
        3) if still running -> proc.kill()

    Args:
        proc: The asyncio subprocess handle.

    Returns:
        None.
    """

    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=3)
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


async def run_training_subprocess(
        cmd: list[str],
        request: Request | None = None,
) -> Tuple[int, str, str]:
    """"
    Run the training worker subprocess and optionally stop early on client disconnect.

    If request is provided, the function races:
        - worker completion (proc.communicate)
        - client disconnect (_watch_disconnect)

    Disconnect behavior:
        - terminates/kill the worker subprocess
        - cancels pending communicate task
        - raises UserDisconnectedException

    Args:
        cmd: Subprocess argv (as produced by build_train_worker_cmd()).
        request: Optional FastAPI Request used to detect disconnect and stop training early.

    Returns:
        A tuple (return_code, stdout_text, stderr_text).

    Raises:
        UserDisconnectedException: If the client disconnects before the worker completes.
        asyncio.CancelledError: If the server cancels the handler task; the worker is terminated.
    """

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    comm_task: asyncio.Task | None = None
    watch_task: asyncio.Task | None = None

    try:
        comm_task = asyncio.create_task(proc.communicate())

        # No request: just wait for worker completion
        if request is None:
            out_b, err_b = await comm_task
            return (
                proc.returncode,
                out_b.decode() if out_b else "",
                err_b.decode() if err_b else "",
            )

        # With request: race disconnect vs worker completion
        watch_task = asyncio.create_task(_watch_disconnect(request))

        done, _ = await asyncio.wait(
            {comm_task, watch_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # 1) Disconnect finished first -> kill worker -> raise
        if watch_task in done:
            await _terminate_proc(proc)

            if comm_task and not comm_task.done():
                comm_task.cancel()
                with suppress(asyncio.CancelledError):
                    await comm_task

            raise UserDisconnectedException()

        # 2) Worker finished first -> cancel watcher -> return output
        if watch_task and not watch_task.done():
            watch_task.cancel()
            with suppress(asyncio.CancelledError):
                await watch_task

        out_b, err_b = comm_task.result()
        return (
            proc.returncode,
            out_b.decode() if out_b else "",
            err_b.decode() if err_b else "",
        )

    except asyncio.CancelledError:
        # Server cancelled handler task -> kill worker so it doesn't keep running
        await _terminate_proc(proc)
        raise

    finally:
        # Best-effort task cleanup
        if watch_task is not None and not watch_task.done():
            watch_task.cancel()
            with suppress(asyncio.CancelledError):
                await watch_task
        if comm_task is not None and not comm_task.done():
            comm_task.cancel()
            with suppress(asyncio.CancelledError):
                await comm_task
