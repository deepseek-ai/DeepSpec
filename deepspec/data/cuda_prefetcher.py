from threading import Thread

import torch


def move_batch_to_device(batch, device):
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    # Embedding lookup requires int64; cast on GPU to avoid bloating CPU-to-GPU transfer.
    if moved["input_ids"].dtype != torch.long:
        moved["input_ids"] = moved["input_ids"].to(torch.long)
    return moved


class CUDAPrefetcher:
    """Overlaps DataLoader iteration and H2D transfer with compute.

    Uses a background thread and a dedicated CUDA stream so that both
    next(dataloader) and the H2D copy for the next batch run concurrently
    with the forward/backward of the current batch.
    """

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        self._iter = iter(self.dataloader)
        self._done = False
        self._gpu_batch = None
        self._thread = None
        # First batch: fetch synchronously so __next__ has something to return.
        self._fetch_and_transfer()
        return self

    def _fetch_and_transfer(self):
        """Pop the next CPU batch from the DataLoader and queue H2D on the side stream.

        A collator may return ``None`` when every sample in a micro-batch is
        filtered out (e.g. ConversationCollator dropping samples below
        ``min_loss_tokens``). Skip such batches so downstream consumers never
        observe ``None``.
        """
        while True:
            try:
                cpu_batch = next(self._iter)
            except StopIteration:
                self._done = True
                return
            if cpu_batch is not None:
                break
        with torch.cuda.stream(self.stream):
            self._gpu_batch = move_batch_to_device(cpu_batch, self.device)

    def __next__(self):
        # Join the background thread kicked off in the previous __next__.
        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._done:
            raise StopIteration

        # Make the compute stream wait until the side-stream H2D is complete.
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        batch = self._gpu_batch

        # Prevent the caching allocator from recycling these tensors before
        # the compute stream is done with them.
        for value in batch.values():
            value.record_stream(current)

        # Kick off the next fetch and H2D in a background thread so it
        # overlaps with compute on the batch we are about to return.
        self._thread = Thread(target=self._fetch_and_transfer, daemon=True)
        self._thread.start()

        return batch

    def __len__(self):
        return len(self.dataloader)
