test:
    pytest

test-dist:
    torchrun --nproc_per_node=2 tests/test_distributed_populora.py
