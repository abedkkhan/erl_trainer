import pytest

from erl.memory import ReflectionMemory


def test_memory_starts_empty():
    mem = ReflectionMemory(max_size=10)
    assert len(mem) == 0


def test_add_single_entry():
    mem = ReflectionMemory(max_size=10)
    mem.add("reflection text", "prompt text", reward=0.9)
    assert len(mem) == 1


def test_retrieve_returns_list_of_strings():
    mem = ReflectionMemory(max_size=10)
    mem.add("reflection A", "prompt A", reward=0.8)
    results = mem.retrieve("any prompt", top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_retrieve_most_recent_first():
    mem = ReflectionMemory(max_size=10)
    mem.add("old reflection", "prompt", reward=0.8)
    mem.add("recent reflection", "prompt", reward=0.9)
    results = mem.retrieve("prompt", top_k=2)
    assert results[0] == "recent reflection"
    assert results[1] == "old reflection"


def test_retrieve_top_k_limits_results():
    mem = ReflectionMemory(max_size=10)
    for i in range(6):
        mem.add(f"reflection {i}", "prompt", reward=0.5)
    results = mem.retrieve("prompt", top_k=3)
    assert len(results) == 3


def test_retrieve_fewer_than_top_k_available():
    mem = ReflectionMemory(max_size=10)
    mem.add("only one", "prompt", reward=0.7)
    results = mem.retrieve("prompt", top_k=5)
    assert len(results) == 1
    assert results[0] == "only one"


def test_eviction_removes_oldest_when_full():
    mem = ReflectionMemory(max_size=3)
    mem.add("first", "prompt", reward=0.5)
    mem.add("second", "prompt", reward=0.6)
    mem.add("third", "prompt", reward=0.7)
    mem.add("fourth", "prompt", reward=0.8)

    assert len(mem) == 3
    results = mem.retrieve("prompt", top_k=3)
    assert "first" not in results
    assert "fourth" in results


def test_eviction_maintains_max_size():
    max_size = 5
    mem = ReflectionMemory(max_size=max_size)
    for i in range(20):
        mem.add(f"reflection {i}", "prompt", reward=float(i) / 20)
    assert len(mem) == max_size


def test_clear_resets_memory():
    mem = ReflectionMemory(max_size=10)
    mem.add("reflection A", "prompt", reward=0.9)
    mem.add("reflection B", "prompt", reward=0.8)
    mem.clear()
    assert len(mem) == 0
    assert mem.retrieve("prompt", top_k=5) == []


def test_add_after_clear():
    mem = ReflectionMemory(max_size=3)
    mem.add("old", "prompt", reward=0.5)
    mem.clear()
    mem.add("new", "prompt", reward=0.9)
    assert len(mem) == 1
    assert mem.retrieve("prompt", top_k=1)[0] == "new"


def test_retrieve_empty_memory_returns_empty_list():
    mem = ReflectionMemory(max_size=10)
    results = mem.retrieve("any prompt", top_k=3)
    assert results == []


def test_max_size_one():
    mem = ReflectionMemory(max_size=1)
    mem.add("first", "prompt", reward=0.5)
    mem.add("second", "prompt", reward=0.9)
    assert len(mem) == 1
    assert mem.retrieve("prompt", top_k=1)[0] == "second"
