#!/usr/bin/env python3
"""test37: clay CG workflow with a 10 A model cutoff.

prepare -> train -> generate / export / md -> analyze
The model graph cutoff and training downselection cutoff are both 10 A.
No scalar energy, condition embedding, equilibrium claim or physical clock
calibration is introduced. See TEST37.md for input mapping and limitations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from functools import partial

import ase.io
from ase import Atoms, units
from ase.geometry import find_mic
from ase.neighborlist import primitive_neighbor_list
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parent
DM2_ROOT = Path(os.environ.get("DM2_ROOT", ROOT.parent / "DM2")).expanduser().resolve()
if not (DM2_ROOT / "src/graphite").is_dir():
    raise RuntimeError("Set DM2_ROOT to a checkout containing src/graphite")
sys.path.insert(0, str(DM2_ROOT / "src"))
torch.serialization.add_safe_globals([slice])
from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP
from graphite.transforms import DownselectEdges, RattleParticles

FORMAT = "test37-clay-cg-denoiser-v1"
DATASET_FORMATS = {FORMAT, "test36-clay-cg-denoiser-v1"}
CAVEAT = (
    "test32 sigma-agnostic displacement denoiser, not a scalar energy or a "
    "temperature-conditioned equilibrium score. F=-kBT*dx/sigma_ref^2 is an "
    "experimental approximation. No energy/virial, rigid platelets, explicit "
    "electrostatics, pressure, shear response or physical kinetics are provided. "
    "One model per condition; generation frames are not equilibrium MD data."
)
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def count(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def new_output(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {path}")
    return path


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def rng_state():
    state = {"numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


class InitialEmbedding(nn.Module):
    """Same embedding as test32.py: two species embeddings and Bessel edges."""
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def architecture(num_species, cutoff):
    return dict(num_species=num_species, cutoff_angstrom=cutoff,
                irreps_node_x="8x0e", irreps_node_z="8x0e",
                irreps_hidden="64x0e + 32x1e", irreps_edge="4x0e + 4x1e + 2x2e",
                irreps_out="1x1e", num_convs=3, radial_neurons=[16, 64], num_neighbors=12)


def build_model(config, device):
    values = {k: v for k, v in config.items() if k not in ("num_species", "cutoff_angstrom")}
    return NequIP(init_embed=InitialEmbedding(config["num_species"], config["cutoff_angstrom"]),
                  **values).to(device)


def graph(positions, cell, type_ids, cutoff, device):
    # Same periodic neighbor construction as test32; graph vectors are not
    # differentiable w.r.t. positions. This is deliberate for a dx model.
    i, j, vec = primitive_neighbor_list("ijD", [True] * 3, cell, positions, cutoff=cutoff)
    if not len(i):
        raise ValueError("No graph edges: check box, units and cutoff")
    return Data(x=torch.as_tensor(type_ids, dtype=torch.long, device=device),
                pos=torch.as_tensor(np.asarray(positions).copy(), dtype=torch.float32, device=device),
                edge_index=torch.as_tensor(np.stack((i, j)), dtype=torch.long, device=device),
                edge_attr=torch.as_tensor(vec, dtype=torch.float32, device=device))


def load_dataset(folder):
    folder = Path(folder).resolve()
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("format") not in DATASET_FORMATS or meta.get("length_unit") != "angstrom":
        raise ValueError("Expected test37 dataset with explicit angstrom units")
    pos = np.load(folder / "positions.npy", mmap_mode="r", allow_pickle=False)
    cells = np.load(folder / "cells.npy", mmap_mode="r", allow_pickle=False)
    if pos.shape != (meta["frames"], len(meta["type_ids"]), 3) or cells.shape != (len(pos), 3, 3):
        raise ValueError("Dataset shapes disagree with metadata")
    for name in ("positions.npy", "cells.npy"):
        if digest(folder / name) != meta["sha256"][name]:
            raise ValueError(f"Dataset modified after preparation: {name}")
    return pos, cells, meta


def map_frame(atoms, mapping):
    if not np.all(atoms.pbc) or atoms.cell.volume <= 0 or not np.isfinite(atoms.positions).all():
        raise ValueError("Input must have finite coordinates and a periodic 3D cell")
    if atoms.constraints:
        raise ValueError("Input has constraints; this workflow does not implement rigid platelets")
    mapped, used = [], set()
    for site in mapping["sites"]:
        raw_ids = np.asarray(site["indices"])
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("Mapping indices must be integers")
        ids = raw_ids.astype(int)
        if ids.ndim != 1 or len(ids) == 0 or ids.min() < 0 or ids.max() >= len(atoms):
            raise ValueError("Mapping contains invalid zero-based atom indices")
        if len(set(ids.tolist())) != len(ids) or used.intersection(ids.tolist()):
            raise ValueError("Mapping groups must be disjoint")
        used.update(ids.tolist())
        weights = np.asarray(site.get("weights", np.ones(len(ids)) / len(ids)), dtype=float)
        if (weights.shape != ids.shape or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1)):
            raise ValueError("Mapping weights must be nonnegative and sum to one")
        anchor = atoms.positions[ids[0]]
        displacement, _ = find_mic(atoms.positions[ids] - anchor, atoms.cell, atoms.pbc)
        mapped.append(anchor + weights @ displacement)
    cell = np.asarray(atoms.cell)
    frac = np.asarray(mapped) @ np.linalg.inv(cell)
    return ((frac - np.floor(frac)) @ cell).astype(np.float32), cell


def prepare(args):
    mapping = json.loads(args.mapping.read_text())
    species = mapping["species"]
    if not species or len({s["name"] for s in species}) != len(species):
        raise ValueError("Provide distinct CG species names")
    for s in species:
        positive(s["mass_amu"])
        if not 1 <= int(s["atomic_number"]) <= 118:
            raise ValueError("atomic_number is a representative element for file export")
    if len(mapping["sites"]) < 2:
        raise ValueError("At least two CG sites are required")
    names = [s["name"] for s in species]
    type_ids = [names.index(s["species"]) for s in mapping["sites"]]
    if set(type_ids) != set(range(len(species))):
        raise ValueError("Every declared CG species must occur in the mapping")
    condition = mapping.get("condition")
    if not isinstance(condition, dict) or not condition:
        raise ValueError("mapping.condition must describe this single training condition")
    positive(condition["temperature_k"])
    if args.stride < 1:
        raise ValueError("stride must be positive")
    output = new_output(args.output)
    # Spool first, then construct .npy with mmap: no full trajectory in RAM.
    nframes, identity = 0, None
    with (output / "positions.raw").open("wb") as pf, (output / "cells.raw").open("wb") as cf:
        for frame_index, atoms in enumerate(ase.io.iread(str(args.trajectory), index=":", format=args.input_format)):
            if frame_index % args.stride:
                continue
            current = atoms.numbers.tolist()
            if identity is None:
                identity = current
            if current != identity:
                raise ValueError("Atom count/order/species changed in trajectory")
            pos, cell = map_frame(atoms, mapping)
            pf.write(pos.tobytes())
            cf.write(np.asarray(cell, dtype=np.float64).tobytes())
            nframes += 1
    if nframes < 2:
        raise ValueError("Need at least two trajectory frames")
    for name, dtype, shape in (("positions", np.float32, (nframes, len(type_ids), 3)),
                               ("cells", np.float64, (nframes, 3, 3))):
        raw = np.memmap(output / f"{name}.raw", dtype=dtype, mode="r", shape=shape)
        dst = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for i in range(nframes):
            dst[i] = raw[i]
        dst.flush()
        del dst, raw
        (output / f"{name}.raw").unlink()  # only this invocation's spool, now in .npy
    meta = dict(format=FORMAT, length_unit="angstrom", frames=nframes, species=species,
                type_ids=type_ids, condition=condition, mapping=mapping,
                source_trajectory=str(args.trajectory.resolve()), source_stride=args.stride,
                source_is_equilibrium=bool(mapping.get("source_is_equilibrium", False)),
                scientific_caveat=CAVEAT,
                sha256={name: digest(output / name) for name in ("positions.npy", "cells.npy")})
    save_json(output / "metadata.json", meta)
    print(f"Prepared {nframes} frames, {len(type_ids)} CG sites: {output}")


def checkpoint_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") not in DATASET_FORMATS:
        raise ValueError("Expected test37 checkpoint; SiO2 weights are not clay weights")
    model = build_model(ck["architecture"], device)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval(), ck


def train(args):
    positions, cells, meta = load_dataset(args.dataset)
    if len(positions) < 3:
        raise ValueError("Training requires at least three frames")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between zero and 0.5")
    if args.sigma_max < 0.001 or args.large_cutoff < args.cutoff:
        raise ValueError("Require sigma-max >= 0.001 and large-cutoff >= cutoff")
    device = device_for(args.device)
    if device.type == "cuda":
        # Safe CUDA-only throughput settings for Ampere/Hopper GPUs. The
        # model remains float32; TF32 accelerates matrix products without
        # changing checkpoint compatibility.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = args.output.resolve() if args.resume else new_output(args.output)
    checkpoint = output / "checkpoint.pt"
    if args.resume and not checkpoint.is_file():
        raise ValueError("--resume requires output/checkpoint.pt")
    # Split real frames BEFORE replication/noising (test32 duplicates first).
    split = max(1, int(len(positions) * (1 - args.validation_fraction)))
    settings = dict(dataset_sha256=meta["sha256"], metadata_sha256=digest(args.dataset / "metadata.json"),
                    cutoff=args.cutoff, large_cutoff=args.large_cutoff, sigma_max=args.sigma_max,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    seed=args.seed, split_frame=split, device=str(device), log_every=args.log_every)
    config = architecture(len(meta["species"]), args.cutoff)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history, completed = [], 0
    if args.resume:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck.get("format") != FORMAT or ck["settings"] != settings:
            raise ValueError("Resume settings/data/device differ from checkpoint")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        history, completed = ck["history"], ck["completed_updates"]
    rattle, downselect = RattleParticles(sigma_max=args.sigma_max), DownselectEdges(cutoff=args.cutoff)
    deadline = time.monotonic() + args.time_budget_hours * 3600

    def save():
        save_checkpoint(checkpoint, dict(format=FORMAT, architecture=config, settings=settings,
            model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), rng=rng_state(), history=history,
            completed_updates=completed, requested_updates=args.updates,
            dataset_metadata=meta, large_cutoff=args.large_cutoff,
            start_positions_angstrom=np.asarray(positions[0]).copy(),
            cell_angstrom=np.asarray(cells[0]).copy(), scientific_caveat=CAVEAT))
        save_json(output / "training.json", dict(completed_updates=completed,
            requested_updates=args.updates, history=history, scientific_caveat=CAVEAT))

    print(f"train: device={device}, frames={len(positions)}, train/validation={split}/{len(positions)-split}", flush=True)
    for step in range(completed + 1, args.updates + 1):
        if STOP or time.monotonic() >= deadline:
            save()
            print("Training paused; resume with --resume", flush=True)
            return 75
        model.train()
        indices = np.random.randint(split, size=args.batch_size)
        batch = Batch.from_data_list([graph(positions[i], cells[i], meta["type_ids"],
                                           args.large_cutoff, device) for i in indices])
        batch = downselect(rattle(batch))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(batch), batch.dx)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        completed = step
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            model.eval()
            # Logging must not alter the subsequent training RNG stream.
            saved_rng = rng_state()
            losses = []
            for i in range(split, min(split + 4, len(positions))):
                valid = graph(positions[i], cells[i], meta["type_ids"], args.large_cutoff, device)
                valid = downselect(rattle(valid))
                with torch.no_grad():
                    losses.append(torch.nn.functional.mse_loss(model(valid), valid.dx).item())
            restore_rng(saved_rng)
            val_loss = float(np.mean(losses))
            row = dict(step=step, train_mse_A2=float(loss.detach().cpu()), valid_mse_A2=val_loss)
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_every == 0:
            save()
    save()
    print(f"Checkpoint: {checkpoint}")


def atoms_from_meta(positions, cell, meta):
    ids = np.asarray(meta["type_ids"])
    atoms = Atoms(numbers=[meta["species"][i]["atomic_number"] for i in ids],
                  positions=positions, cell=cell, pbc=True,
                  masses=[meta["species"][i]["mass_amu"] for i in ids])
    atoms.set_array("cg_type", ids.copy())
    return atoms


@torch.no_grad()
def generate(args):
    if args.max_sigma < 0.001:
        raise ValueError("max-sigma must be at least 0.001 angstrom")
    device = device_for(args.device)
    model, ck = checkpoint_model(args.checkpoint, device)
    meta, cell = ck["dataset_metadata"], ck["cell_angstrom"]
    cutoff = ck["architecture"]["cutoff_angstrom"]
    output = args.output.resolve() if args.resume else new_output(args.output)
    settings = dict(checkpoint_sha256=digest(args.checkpoint), noisy_steps=args.noisy_steps,
                    polish_steps=args.polish_steps, max_sigma=args.max_sigma,
                    seed=args.seed, device=str(device), cutoff_angstrom=cutoff)
    total = args.noisy_steps + args.polish_steps
    state_path = output / "generation_restart.pt"
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("Generation resume settings differ (including graph cutoff); "
                             "use a new output directory for legacy generation runs")
        pos, completed = state["positions"].to(device), state["step"]
        restore_rng(state["rng"])
        trajectory = np.lib.format.open_memmap(output / "positions.npy", mode="r+")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        pos = torch.tensor(ck["start_positions_angstrom"], dtype=torch.float32, device=device)
        completed = 0
        trajectory = np.lib.format.open_memmap(output / "positions.npy", mode="w+", dtype=np.float32,
                                             shape=(total + 1, len(pos), 3))
        trajectory[0] = pos.cpu().numpy()
    sigmas = torch.linspace(args.max_sigma, 0.001, args.noisy_steps, device=device)
    deadline = time.monotonic() + args.time_budget_hours * 3600

    def save():
        trajectory.flush()
        save_checkpoint(state_path, dict(settings=settings, step=completed,
                        positions=pos.detach().cpu(), rng=rng_state()))
        save_json(output / "generation.json", dict(completed_steps=completed, requested_steps=total,
                  valid_frames=completed + 1, complete=completed == total, length_unit="angstrom",
                  settings=settings,
                  cell_angstrom=np.asarray(cell).tolist(), dataset_metadata=meta,
                  is_equilibrium_trajectory=False, scientific_caveat=CAVEAT))

    for step in range(completed, total):
        if STOP or time.monotonic() >= deadline:
            save()
            print("Generation paused; resume with --resume")
            return 75
        # Training's large_cutoff is only a candidate-neighbor buffer before
        # rattling/downselection. The model must see the training cutoff here.
        data = graph(pos.cpu().numpy(), cell, meta["type_ids"], cutoff, device)
        # Exact test32 generation ordering: evaluate current coordinates, then
        # subtract model(data) + sigma*noise. No new reverse SDE is substituted.
        displacement = model(data)
        if step < args.noisy_steps:
            displacement = displacement + sigmas[step] * torch.randn_like(pos)
        pos = pos - displacement
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite generated positions")
        completed = step + 1
        trajectory[completed] = pos.cpu().numpy()
        if completed % args.checkpoint_every == 0:
            save()
            print(f"generation {completed}/{total}", flush=True)
    save()
    atoms = atoms_from_meta(pos.cpu().numpy(), cell, meta)
    atoms.wrap()
    ase.io.write(output / "final.extxyz", atoms)
    print(f"Generated {total + 1} frames: {output}")


def export(args):
    _, ck = checkpoint_model(args.checkpoint, torch.device("cpu"))
    meta, cell = ck["dataset_metadata"], np.asarray(ck["cell_angstrom"])
    if not np.allclose(cell, np.diag(np.diag(cell))):
        raise ValueError("Existing LAMMPS callback supports orthorhombic fixed boxes only")
    output = new_output(args.output)
    atoms = atoms_from_meta(ck["start_positions_angstrom"], cell, meta)
    atoms.wrap()
    lines = ["test37 CG start (species order in bundle JSON)", "", f"{len(atoms)} atoms",
             f"{len(meta['species'])} atom types", ""]
    lines += [f"0 {length:.16g} {axis}lo {axis}hi" for length, axis in zip(np.diag(cell), "xyz")]
    lines += ["", "Masses", ""]
    lines += [f"{i+1} {s['mass_amu']}" for i, s in enumerate(meta["species"])]
    lines += ["", "Atoms # atomic", ""]
    lines += [f"{i+1} {meta['type_ids'][i]+1} {p[0]:.16g} {p[1]:.16g} {p[2]:.16g}"
              for i, p in enumerate(atoms.positions)]
    (output / "cg_start.data").write_text("\n".join(lines) + "\n")
    bundle = dict(format="test32-score-forcefield", format_version=1, kind="coarse_grained",
                  architecture=ck["architecture"], model_state_dict=ck["model_state_dict"],
                  species_by_id=torch.tensor(meta["type_ids"]),
                  atomic_numbers_by_id=torch.tensor(atoms.numbers), num_atoms=len(atoms),
                  cell_angstrom=torch.tensor(cell), species=meta["species"], condition=meta["condition"],
                  temperature_k=float(meta["condition"]["temperature_k"]), sigma_ref_angstrom=args.sigma_ref,
                  force_clip_ev_per_angstrom=args.force_clip, suggested_timestep_ps=0.0001,
                  suggested_damping_ps=0.1, suggested_data_file="cg_start.data",
                  conservative=False, provides_energy=False, provides_virial=False,
                  scientific_status="experimental_unvalidated", scientific_caveat=CAVEAT,
                  source_checkpoint_sha256=digest(args.checkpoint))
    save_checkpoint(output / "cg_score_forcefield.pt", bundle)
    public = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in bundle.items() if k != "model_state_dict"}
    save_json(output / "cg_score_forcefield.json", public)
    print(f"LAMMPS force bundle: {output}")


def md(args):
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    device = device_for(args.device)
    model, ck = checkpoint_model(args.checkpoint, device)
    meta, cell = ck["dataset_metadata"], ck["cell_angstrom"]
    temperature = float(meta["condition"]["temperature_k"])
    output = new_output(args.output)
    atoms = atoms_from_meta(ck["start_positions_angstrom"], cell, meta)

    class ScoreCalculator(Calculator):
        implemented_properties = ["forces"]

        def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            with torch.no_grad():
                data = graph(self.atoms.positions, np.asarray(self.atoms.cell), meta["type_ids"],
                             ck["architecture"]["cutoff_angstrom"], device)
                force = -units.kB * temperature * model(data) / args.sigma_ref**2
                force -= force.mean(dim=0, keepdim=True)
                force *= (args.force_clip / force.norm(dim=1, keepdim=True).clamp_min(1.e-12)).clamp(max=1)
                force -= force.mean(dim=0, keepdim=True)
                if not torch.isfinite(force).all():
                    raise RuntimeError("Non-finite score force")
                self.results["forces"] = force.cpu().numpy().astype(float)

    atoms.calc = ScoreCalculator()
    atoms.set_constraint(FixCom())
    random = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=random)
    Stationary(atoms)
    dyn = Langevin(atoms, timestep=args.timestep_fs * units.fs, temperature_K=temperature,
                   friction=1 / (args.damping_ps * 1000 * units.fs), rng=random, fixcm=False)
    started = time.monotonic()
    completed = 0
    with (output / "md.extxyz").open("w") as trajectory, \
            (output / "thermo.csv").open("w", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_ps", "temperature_k", "max_force_eV_A", "elapsed_seconds"])

        def record():
            ase.io.write(trajectory, atoms, format="extxyz", write_results=False)
            trajectory.flush()
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            elapsed = time.monotonic() - started
            writer.writerow([completed, completed * args.timestep_fs / 1000, atoms.get_temperature(), fmax, elapsed])
            save_json(output / "progress.json", dict(completed_steps=completed, requested_steps=args.steps,
                      elapsed_seconds=elapsed, temperature_k=atoms.get_temperature()))
            ase.io.write(output / "latest.extxyz", atoms, write_results=False)
            print(f"MD {completed}/{args.steps}: T={atoms.get_temperature():.2f} K", flush=True)

        record()
        while completed < args.steps:
            if STOP or time.monotonic() - started >= args.time_budget_hours * 3600:
                print("MD paused; latest.extxyz has positions and velocities, not an exact RNG restart")
                return 75
            increment = min(args.save_every, args.steps - completed)
            dyn.run(increment)
            completed += increment
            record()
    ase.io.write(output / "final.extxyz", atoms, write_results=False)
    save_json(output / "run_metrics.json", dict(completed_steps=completed, time_ps=completed * args.timestep_fs / 1000,
              temperature_k=temperature, sigma_ref_angstrom=args.sigma_ref,
              force_clip_eV_A=args.force_clip, timestep_fs=args.timestep_fs, damping_ps=args.damping_ps,
              seed=args.seed, device=str(device), elapsed_seconds=time.monotonic()-started,
              checkpoint_sha256=digest(args.checkpoint), scientific_caveat=CAVEAT))


def analyze(args):
    positions, cells, meta = load_dataset(args.reference)
    if args.samples.is_dir():
        state = json.loads((args.samples / "generation.json").read_text())
        if (state["dataset_metadata"]["type_ids"] != meta["type_ids"]
                or state["dataset_metadata"]["species"] != meta["species"]
                or state["dataset_metadata"]["mapping"] != meta["mapping"]):
            raise ValueError("Generated species/mapping differ from reference")
        generated = np.load(args.samples / "positions.npy", mmap_mode="r", allow_pickle=False)
        samples = [(generated[i], np.array(state["cell_angstrom"])) for i in range(0, state["valid_frames"], args.stride)]
        source_kind = "denoising_generation_not_equilibrium"
    else:
        samples = []
        for i, atoms in enumerate(ase.io.iread(str(args.samples), index=":")):
            if i % args.stride == 0:
                if len(atoms) != len(meta["type_ids"]) or not np.array_equal(atoms.arrays.get("cg_type"), meta["type_ids"]):
                    raise ValueError("MD trajectory CG types/order differ from reference")
                samples.append((atoms.positions, np.asarray(atoms.cell)))
        source_kind = "score_md"
    if not samples:
        raise ValueError("No samples to analyze")
    output = new_output(args.output)
    types = np.asarray(meta["type_ids"])
    ntypes = len(meta["species"])
    edges = np.linspace(0, args.r_max, 101)
    shell = 4 * np.pi / 3 * np.diff(edges**3)
    pairs = [(i, j) for i in range(ntypes) for j in range(i, ntypes)]

    def hist(pos, cell):
        heights = 1 / np.linalg.norm(np.linalg.inv(cell), axis=0)
        if args.r_max > heights.min() / 2:
            raise ValueError("RDF r-max exceeds half the shortest cell height")
        i, j, distances = primitive_neighbor_list("ijd", [True]*3, cell, pos, cutoff=args.r_max)
        out = {}
        for a, b in pairs:
            na, nb = np.sum(types == a), np.sum(types == b)
            denominator = na * (nb - int(a == b)) * shell / np.linalg.det(cell)
            if na * (nb - int(a == b)) == 0:
                continue
            selected = distances[(types[i] == a) & (types[j] == b)]
            out[f"{meta['species'][a]['name']}--{meta['species'][b]['name']}"] = np.histogram(selected, edges)[0] / denominator
        return out

    reference = [hist(positions[i], cells[i]) for i in range(0, len(positions), args.stride)]
    sampled = [hist(pos, cell) for pos, cell in samples]
    curves = {key: dict(reference=np.mean([h[key] for h in reference], axis=0).tolist(),
                       sampled=np.mean([h[key] for h in sampled], axis=0).tolist()) for key in reference[0]}
    report = dict(r_angstrom=((edges[1:]+edges[:-1])/2).tolist(), partial_rdf=curves,
                  reference_frames=len(reference), sample_frames=len(samples), sample_kind=source_kind,
                  equilibrium_validated=False, unsupported_observables=["swelling_pressure", "physical_transport", "shear_stress", "tactoid_size", "pore_size", "d001"],
                  scientific_caveat=CAVEAT)
    save_json(output / "rdf.json", report)
    print(f"Partial RDF: {output / 'rdf.json'}")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare", help="Map external atomistic trajectory to CG sites")
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--input-format", default=None)
    p.add_argument("--stride", type=count, default=1)
    p.set_defaults(handler=prepare)
    p = sub.add_parser("train", help="test32 NequIP + RattleParticles + displacement MSE")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--updates", type=count, default=30000)
    p.add_argument("--batch-size", type=count, default=16)
    p.add_argument("--learning-rate", type=positive, default=2.e-4)
    p.add_argument("--cutoff", type=positive, default=10.0)
    p.add_argument("--large-cutoff", type=positive, default=10.0)
    p.add_argument("--sigma-max", type=positive, default=0.75)
    p.add_argument("--validation-fraction", type=positive, default=0.1)
    p.add_argument("--log-every", type=count, default=100)
    p.set_defaults(handler=train)
    p = sub.add_parser("generate", help="Exact test32 annealed updates and polish")
    p.add_argument("--noisy-steps", type=count, default=2900)
    p.add_argument("--polish-steps", type=count, default=100)
    p.add_argument("--max-sigma", type=positive, default=1.0)
    p.set_defaults(handler=generate)
    p = sub.add_parser("export", help="Bundle for existing test32_lammps.py callback")
    p.set_defaults(handler=export)
    p = sub.add_parser("md", help="Experimental fixed-cell ASE Langevin score MD")
    p.add_argument("--steps", type=count, default=1000)
    p.add_argument("--timestep-fs", type=positive, default=0.1)
    p.add_argument("--damping-ps", type=positive, default=0.1)
    p.add_argument("--save-every", type=count, default=100)
    p.set_defaults(handler=md)
    p = sub.add_parser("analyze", help="Partial RDFs by CG species")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--r-max", type=positive, default=5.0)
    p.add_argument("--stride", type=count, default=1)
    p.set_defaults(handler=analyze)
    for name, p in sub.choices.items():
        p.add_argument("--output", type=Path, required=True)
        if name in ("generate", "export", "md"):
            p.add_argument("--checkpoint", type=Path, required=True)
        if name in ("train", "generate", "md"):
            p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
            p.add_argument("--seed", type=int, default=1337)
            p.add_argument("--time-budget-hours", type=positive, default=19.5)
        if name in ("train", "generate"):
            p.add_argument("--resume", action="store_true")
            p.add_argument("--checkpoint-every", type=count, default=100)
        if name in ("export", "md"):
            p.add_argument("--sigma-ref", type=positive, required=True)
            p.add_argument("--force-clip", type=positive, default=10.0)
    return root


def main():
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args = parser().parse_args()
    print(CAVEAT, flush=True)
    return args.handler(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
