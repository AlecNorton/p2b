# Install

## Planning (no GPU needed)

```bash
conda create -n aerial_robotics python=3.10
conda activate aerial_robotics
mkdir -p ~/rbe595 && cd ~/rbe595
git clone -b dev https://github.com/pearwpi/splat_hitl.git
pip install -e splat_hitl
```

Use `-b dev`: `main` cannot load this pack's contract. Put this starter pack in
`~/rbe595` too, and check the setup from it:

```bash
python3 tools/hover_check.py
```

It stops at the first problem. The flight stack, `cf_vicon_stack`, is already
on the lab PC.

## GPU rendering (optional, for the §5.4 videos)

On [Turing](https://docs.turing.wpi.edu/), in the same environment:

```bash
module load cuda/12.4.0/3mdaov5      # after every login
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install gsplat
```

The Turing tutorial video on the assignment page shows the CUDA setup.
`splat_hitl/worker/README.md` shows how to render. Without a GPU,
`tools/splat_cpu.py` renders with numpy, slowly.

## Problems

| symptom | fix |
|---|---|
| `ModuleNotFoundError: splat_hitl` | `pip install -e splat_hitl`, in the active environment |
| the bundle loads but the contract does not | you are on `main`: `git -C splat_hitl fetch origin && git -C splat_hitl checkout dev` |
| `check()` prints two warnings | expected: see `scene/README.md` |
| `torch.cuda.is_available()` is `False` on Turing | run on a GPU node, after `module load cuda/12.4.0/3mdaov5` |
