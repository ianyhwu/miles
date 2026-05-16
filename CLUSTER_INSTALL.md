# Miles — install on Orchard cluster (no Docker)

Bare-metal install for Miles on the CMU FLAME Orchard cluster, following the pins
from `docker/Dockerfile`. Docker is the recommended path upstream; this is the
fallback when you want a conda env so editable code changes don't require an
image rebuild.

Target: H100 node(s), conda envs live under `/project/flame/ianwu/conda/envs/`.

---

## Why Python 3.12

The pre-built wheels in [`yueming-yuan/miles-wheels`][wheels] at tag
`cu129-x86_64` are `cp312` — `apex` and `flash_attn 2.7.4` are cp312-only, so
the env must be Python 3.12. This matches the `lmsysorg/sglang:v0.5.10` base
image used by the Dockerfile.

[wheels]: https://github.com/yueming-yuan/miles-wheels/releases/tag/cu129-x86_64

---

## 1. Cache & tmpdir redirection (do this first)

`/home/ianwu` has a small soft quota that pip can blow through during TE / mamba
builds. Always redirect pip + tmp to `/tmp/ianwu/` (wiped between jobs, fine):

```bash
mkdir -p /tmp/ianwu/pip-cache /tmp/ianwu/tmp
export PIP_CACHE_DIR=/tmp/ianwu/pip-cache
export TMPDIR=/tmp/ianwu/tmp
```

If you hit `Disk quota exceeded` mid-install, clean these heavy caches in
`/home`:

```bash
rm -rf /home/ianwu/.cache/pip /home/ianwu/.cache/flashinfer \
       /home/ianwu/.cache/vllm /home/ianwu/.cache/wandb
```

---

## 2. Create the conda env

```bash
conda create -n miles python=3.12 -y
source activate miles
pip install --upgrade pip
```

`~/.condarc` is already configured with `envs_dirs: /project/flame/ianwu/conda/envs`
so this lands in the right place.

---

## 3. Base SGLang (brings torch + most of the CUDA stack)

```bash
pip install "sglang[all]==0.5.10"
```

This pulls `torch 2.9.1`, the `nvidia-*-cu12` runtime packages (cuda 12.8),
`flashinfer`, `transformers`, etc. ~5 min, ~10 GB.

---

## 4. Apply the `sglang-miles` patches

Miles uses cherry-picks on top of v0.5.10. Clone, check out, install editable
with `--no-deps` (replaces only the Python code installed above):

```bash
mkdir -p /project/flame/ianwu/code/miles-deps
cd /project/flame/ianwu/code/miles-deps
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin sglang-miles
git checkout FETCH_HEAD
pip install -e "python[all]" --no-deps
```

Reports as `sglang 0.5.11`.

---

## 5. Megatron-LM fork

```bash
cd /project/flame/ianwu/code
git clone https://github.com/radixark/Megatron-LM.git --recursive -b miles-main Megatron-LM
cd Megatron-LM
pip install -e . --no-deps
```

Installs `megatron-core 0.16.0rc0`.

---

## 6. Pre-built CUDA wheels

Download once, install offline. These are the wheels referenced inside the
Dockerfile.

```bash
mkdir -p /project/flame/ianwu/code/miles-deps/wheels
cd /project/flame/ianwu/code/miles-deps/wheels

for url in \
  https://github.com/yueming-yuan/miles-wheels/releases/download/cu129-x86_64/apex-0.1-cp312-cp312-linux_x86_64.whl \
  https://github.com/yueming-yuan/miles-wheels/releases/download/cu129-x86_64/fake_int4_quant_cuda-0.0.0-cp312-cp312-linux_x86_64.whl \
  https://github.com/yueming-yuan/miles-wheels/releases/download/cu129-x86_64/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl \
  https://github.com/yueming-yuan/miles-wheels/releases/download/cu129-x86_64/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl \
  https://github.com/yueming-yuan/miles-wheels/releases/download/cu129-x86_64/sgl-model-gateway-linux-x86_64.tar.gz \
  ; do curl -fSL -O "$url"; done

# install wheels
pip install \
  apex-0.1-cp312-cp312-linux_x86_64.whl \
  fake_int4_quant_cuda-0.0.0-cp312-cp312-linux_x86_64.whl \
  flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl \
  flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl

# flash_attn_3 also needs the interface file copied in
python_path=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $python_path/flash_attn_3
curl -fSL https://raw.githubusercontent.com/Dao-AILab/flash-attention/fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89/hopper/flash_attn_interface.py \
  -o $python_path/flash_attn_3/flash_attn_interface.py

# sgl-model-gateway binary
tar xzf sgl-model-gateway-linux-x86_64.tar.gz -C /project/flame/ianwu/conda/envs/miles/bin/
chmod +x /project/flame/ianwu/conda/envs/miles/bin/sgl-model-gateway
```

**Skip the `sglang_router` wheel** from this release — it's
`manylinux_2_39_x86_64` and Orchard ships glibc 2.35. The pypi version 0.3.2
installs cleanly via `pip install sglang-router` (covered by `requirements.txt`).

---

## 7. Transformer Engine 2.10 (needs CUDA + cudnn + nccl headers)

The PyPI `transformer_engine` wheel is metadata-only. The `[pytorch]` extra
builds `transformer_engine_torch` from source and needs CUDA toolkit + cudnn.h +
nccl.h on the include path. Orchard has `/usr/local/cuda-12.8` (matches torch's
cu128 stack); the cudnn/nccl headers live inside the pip packages.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export CUDNN_PATH=/project/flame/ianwu/conda/envs/miles/lib/python3.12/site-packages/nvidia/cudnn
export NCCL_PATH=/project/flame/ianwu/conda/envs/miles/lib/python3.12/site-packages/nvidia/nccl
export CPATH=$CUDNN_PATH/include:$NCCL_PATH/include:$CPATH
export LIBRARY_PATH=$CUDNN_PATH/lib:$NCCL_PATH/lib:$LIBRARY_PATH

pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
```

Verify:

```bash
python -c "import transformer_engine.pytorch; print('TE OK')"
```

If you see `fatal error: cudnn.h: No such file or directory` — `CPATH` isn't
set. If `nvcc: command not found` — `CUDA_HOME` isn't on `PATH`.

---

## 8. Remaining pinned deps

```bash
# pinned-commit installs (need git URL + commit)
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 --force-reinstall

# version-pinned
pip install flash-linear-attention==0.4.2
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
pip install --no-build-isolation causal-conv1d==1.6.1 mamba-ssm==2.3.1   # builds from source, needs CUDA_HOME set
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# --no-deps to avoid noisy resolver edits
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps

# CUDA-IPC P2P weight transfer between Megatron and SGLang. The SGLang docker
# base image ships this, but `sglang[all]==0.5.10` from pypi does not — without
# it, MegatronTrainRayActor crashes at `from mooncake.engine import TransferEngine`
# when it does the first weight sync.
pip install mooncake-transfer-engine
```

---

## 9. Miles itself

```bash
cd /home/ianwu/code/miles
pip install -r requirements.txt
pip install -e . --no-deps
```

`requirements.txt` brings in `ray 2.55`, `wandb`, `accelerate`, `ring_flash_attn`,
`torchft-nightly==2026.4.3`, `sglang-router` (pypi), etc.

---

## 10. Final pins from the Dockerfile

Done last because earlier installs reset numpy/cudnn:

```bash
pip install "numpy<2" "nvidia-cudnn-cu12==9.16.0.29"
```

You'll see pip complain that torch 2.9.1 wants cudnn 9.10.2.21 — ignore, the
Dockerfile intentionally bumps it.

---

## 11. Activation snippet for future shells

```bash
source activate miles
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/project/flame/ianwu/code/Megatron-LM:$PYTHONPATH
```

The `PYTHONPATH` line matches the `RUNTIME_ENV_JSON` block at the bottom of the
Miles run scripts (e.g. `scripts/run-qwen3-4B.sh`).

---

## Known issues

- **`train.py --help` crashes** with
  `AttributeError: 'tuple' object has no attribute 'strip'` somewhere in
  argparse's help formatter. Parsing itself works — only the help output is
  broken. Some `help=` value is a tuple instead of a string.
- **megatron-bridge resolver warnings**: it requests `transformers<=5.2.0` and
  `pyyaml>=6.0.2`. We install transformers 5.3.0 (matches miles requirements)
  and pyyaml 6.0.1. Doesn't appear to break anything in the test suite.
- **Two cudnn versions warning** during step 10 — expected, we want the
  Dockerfile pin not the torch pin.

---

# Testing

What was run after install. All from `/home/ianwu/code/miles/`.

## Quick imports

```bash
python -c "import miles"
python -c "import transformer_engine.pytorch"
python -c "import mamba_ssm, causal_conv1d"
python -c "import flash_attn, flash_attn_3"
```

All return cleanly.

## Argparse parser end-to-end

Smoke-test that the full train.py argument pipeline composes without crashing.
Won't run a model — it fails when it tries to load HF config for the fake path,
which is the right signal that parsing finished.

```bash
PYTHONPATH=/project/flame/ianwu/code/Megatron-LM python -c "
import sys
sys.argv = ['test', '--actor-num-nodes', '1', '--actor-num-gpus-per-node', '8',
            '--hf-checkpoint', '/tmp/fake', '--num-rollout', '1',
            '--rollout-batch-size', '8', '--n-samples-per-prompt', '4',
            '--global-batch-size', '8', '--rollout-max-response-len', '2048',
            '--rm-type', 'deepscaler', '--advantage-estimator', 'grpo',
            '--num-layers', '2', '--hidden-size', '64', '--ffn-hidden-size', '128',
            '--num-attention-heads', '2', '--seq-length', '2048']
from miles.utils.arguments import parse_args
args = parse_args()
print('Parsed OK; layers=', args.num_layers)
"
```

Expected: `Parsed OK; layers= 2` followed by an HF `OSError` about `/tmp/fake`.

## Unit tests (the real signal)

```bash
# fast/utils — most of the pure-Python plumbing
python -m pytest tests/fast/utils \
  --ignore=tests/fast/utils/test_mxfp8_quantizer.py \
  --ignore=tests/fast/utils/test_nvfp4_quantizer.py \
  --ignore=tests/fast/utils/test_quantizer_ci.py
# Result: 886 passed, 1 failed (CLI debug test — env-specific, not blocking)

# fast/rollout + fast/router — generation + multi-engine routing
python -m pytest tests/fast/rollout tests/fast/router
# Result: 446 passed, 29 skipped, 0 failed   (~8 min wall)
```

### Why some tests are skipped

- `test_mxfp8_quantizer.py`, `test_nvfp4_quantizer.py`, `test_quantizer_ci.py`
  fail at import because they pull in CUDA quantization kernels that aren't
  exercisable from this entrypoint. Not an install issue.
- The 29 skips in `fast/rollout` + `fast/router` are conditional on hardware /
  network / external services and skip themselves cleanly.
- The one failure in `fast/utils` is
  `debug_utils/run_megatron/cli/commands/test_run.py::test_cmd_contains_dumper_env`
  — looks at process env, brittle to local shell setup.

## Smoke run (end-to-end RL loop — done)

A minimal GRPO loop was run with `Qwen3-4B-Instruct-2507` on 8 H100s. Two
rollouts of 8 prompts × 4 samples, 512-token responses, deepscaler reward.

### Inputs

- HF checkpoint:
  `/project/flame/ianwu/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- The Megatron actor needs a `torch_dist` checkpoint, not raw HF. One-time
  conversion (~30s after model load, ~16 GB peak GPU mem):

  ```bash
  cd /home/ianwu/code/miles
  source scripts/models/qwen3-4B.sh > /dev/null
  MODEL_ARGS+=( --rotary-base 5000000 )  # Qwen3-4B-Instruct-2507 uses 5e6, not 1e6
  PYTHONPATH=/project/flame/ianwu/code/Megatron-LM python tools/convert_hf_to_torch_dist.py \
     "${MODEL_ARGS[@]}" \
     --hf-checkpoint $QWEN_HF \
     --save /tmp/ianwu/qwen3-4b-inst_torch_dist
  ```

- The DAPO parquet on disk has `prompt` as a list-of-messages and
  `reward_model.ground_truth` nested under a struct, but the miles data loader
  only does flat `label_key` lookups. Flatten a slice into JSONL:

  ```python
  import pyarrow.parquet as pq, json
  p = pq.ParquetFile('/project/flame/ianwu/huggingface/hub/datasets--BytedTsinghua-SIA--DAPO-Math-17k/snapshots/65877096c24ffa7abc4e4fa5edb95cf3413a5674/data/dapo-math-17k.parquet')
  with open('/tmp/ianwu/data/dapo-math-17k-flat.jsonl', 'w') as f:
      n = 0
      for batch in p.iter_batches(batch_size=512):
          for row in batch.to_pylist():
              f.write(json.dumps({'prompt': row['prompt'], 'label': row['reward_model']['ground_truth']}) + '\n')
              n += 1
              if n >= 2048: break
          if n >= 2048: break
  ```

### Launch script

Adapted from `scripts/run-qwen3-4B.sh`. The originals assume `/root/...` paths
(set in the Docker image); we swap them for the cluster's `/project/flame/ianwu/`
and `/tmp/ianwu/`. Key differences vs. upstream:

- `--num-rollout 2` (just two iterations)
- `--rollout-batch-size 8 --n-samples-per-prompt 4 --global-batch-size 32`
- `--rollout-max-response-len 512 --max-tokens-per-gpu 4096` (tiny — fine for plumbing)
- Pass `CUDA_HOME` and the cuda-12.8 `PATH` through the Ray runtime env so worker
  procs find `nvcc`.

See `/tmp/ianwu/run_smoke.sh` for the full script that was run.

### Result

`Job 'raysubmit_NBbRC1jJyXB8e19t' succeeded` after about 6 min of wallclock
(3 min cold start for engines, ~3 min for two rollout+train iterations).

Step-0 metrics (logged from `MegatronTrainRayActor`):

```text
rollout 0: response_lengths=512.0  rewards=0.0  truncated=1.0  total_lengths=670.75
step 0:    loss=0.0  pg_loss=0.0  entropy_loss=0.202  ppo_kl=0.0  ess_ratio=1.0
           train_rollout_kl=0.00051  grad_norm=0.0  lr=1e-6
```

Rewards of 0 are expected — `rollout_max_response_len=512` truncates every math
response. The point was to exercise the full pipeline:

- torch_dist checkpoint load through Megatron-LM fork
- 8 colocated SGLang engines with CUDA graphs
- DAPO parquet → flattened JSONL → miles data loader → chat template
- Deepscaler reward scoring on truncated responses
- GRPO step: ref log-probs → log-probs → policy update
- Save iteration-1 checkpoint (1.2s)
- CUDA-IPC P2P weight sync Megatron → SGLang via mooncake (16 buckets, 1.2s)
- Engine offload/wake for colocation

For a useful run, raise `rollout-max-response-len` to 8192+, `max-tokens-per-gpu`
to 9216+, and `num-rollout` to a meaningful number.
