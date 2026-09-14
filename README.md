# test37

Clay coarse-grained denoising workflow derived from test36.

This version uses a 10 Å model/training cutoff and a 20-hour PBS job. The
training stage has a 19.5-hour budget, saves checkpoints, and can be resumed
with `RESUME=1`.

## Supercomputer quick start

```bash
git clone https://github.com/haru2225/test37.git
cd test37
singularity build --fakeroot test37.sif Singularity.test37.def
qsub -P <ProjectGroup_ID> \
  -v AA_RUN=/path/to/aa-run,DM2_ROOT=/path/to/DM2,SIF_IMAGE=$PWD/test37.sif \
  run_test37.pbs
```

Set `AA_RUN` to the directory containing `production.extxyz` and `mapping.json`.
See `TEST37.md` for restart and output details.
