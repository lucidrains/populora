test:
    uv run --extra test pytest

test-dist:
    uv run --extra test torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 tests/test_distributed_populora.py

smoke-mario-dist:
    uv run --extra mario torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=4 validate_with_mario.py --pop_size 32 --max_generations 3 --max_stagnant_steps 60 --render_video False --video_dir /tmp/mario-dist-smoke --checkpoint_dir /tmp/mario-dist-smoke-ckpt
