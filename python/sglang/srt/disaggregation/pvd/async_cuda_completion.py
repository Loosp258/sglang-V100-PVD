"""Local CUDA events only; this is not a GPUDirect remote-write ordering proof."""
import asyncio
import threading
import torch


async def wait_local_cuda_event(event, device):
    """Yield on the original owner until completion; cancellation still joins it."""
    owner=threading.get_ident()
    cancelled=None
    while True:
        if threading.get_ident()!=owner:
            raise RuntimeError('CUDA event completion moved to another owner')
        with torch.cuda.device(device):
            complete=event.query()
        if type(complete) is not bool:
            raise RuntimeError('exact local event completion required')
        if complete:
            if cancelled is not None:raise cancelled
            return
        try:
            await asyncio.sleep(.0005)
        except asyncio.CancelledError as exc:
            cancelled=exc
