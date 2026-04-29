from collections import deque
from dataclasses import dataclass


@dataclass
class _MemoryEntry:
    reflection: str
    prompt: str
    reward: float


class ReflectionMemory:
    """Cross-episode memory storing successful reflections for use in future reflection prompts.

    Entries are stored in insertion order and evicted oldest-first when the capacity is reached.
    Retrieval is recency-based (most recent entries first), suitable as a starting point before
    upgrading to embedding-based similarity retrieval. The ``prompt`` argument to
    :meth:`retrieve` is currently unused and reserved for future similarity-based retrieval.

    .. note::

        This memory is **per-process**. Under multi-GPU / multi-rank training (DDP,
        DeepSpeed, FSDP), each rank maintains its own deque and entries are *not*
        synchronised across ranks. As a consequence, different ranks may sample
        from divergent memory states. If reproducibility across ranks matters,
        either disable memory (``enable_memory=False``) or extend this class to
        gather/broadcast new entries on :meth:`add`.

    Args:
        max_size: Maximum number of reflection entries to store.
    """

    def __init__(self, max_size: int = 50) -> None:
        self._max_size = max_size
        self._entries: deque[_MemoryEntry] = deque()

    def add(self, reflection: str, prompt: str, reward: float) -> None:
        """Store a reflection with its associated prompt and reward.

        If memory is full, the oldest entry is evicted before the new one is added.

        Args:
            reflection: The textual reflection describing what went wrong and how to improve.
            prompt: The original task prompt this reflection is associated with.
            reward: The reward achieved on the second attempt that produced this reflection.
        """
        if len(self._entries) >= self._max_size:
            self._entries.popleft()
        self._entries.append(_MemoryEntry(reflection=reflection, prompt=prompt, reward=reward))

    def retrieve(self, prompt: str, top_k: int = 3) -> list[str]:
        """Retrieve the top_k most recent reflections.

        Args:
            prompt: The current task prompt (reserved for future similarity-based retrieval).
            top_k: Maximum number of reflections to return.

        Returns:
            List of reflection strings, most recent first.
        """
        recent = list(self._entries)[-top_k:]
        return [entry.reflection for entry in reversed(recent)]

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Reset the memory, removing all stored reflections."""
        self._entries.clear()
