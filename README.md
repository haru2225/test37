# test37

Clay coarse-grained denoising workflow derived from test36.

This version uses a 10 Å model/training cutoff and a 20-hour PBS job. The
training stage has a 19.5-hour budget, saves checkpoints, and can be resumed
with `RESUME=1`.

The repository includes the prepared test36 dataset in `input/`. The large
positions array is stored as two GitHub-compatible chunks and reconstructed by
the PBS script. No Git LFS installation is required. The PBS script builds the
container and clones DM2 automatically on the first submission.

## Supercomputer quick start

```bash
git clone https://github.com/haru2225/test37.git
cd test37
qsub -P <ProjectGroup_ID> run_test37.pbs
```

See `TEST37.md` for restart and output details. Set `AA_RUN`, `DM2_ROOT`, or
`SIF_IMAGE` only when replacing the bundled input or site dependencies.
