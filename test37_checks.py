"""Software checks with synthetic coordinates, NOT clay validation data.

Run: ../.pixi/envs/default/bin/python -m unittest test37_checks -v
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import ase.io
from ase import Atoms
import numpy as np
import torch

import test37 as t

warnings.filterwarnings("ignore", message="The TorchScript type system.*")
torch.set_num_threads(1)


class Workflow(unittest.TestCase):
    def setUp(self):
        self.warnings = warnings.catch_warnings()
        self.warnings.__enter__()
        warnings.filterwarnings("ignore", message="The TorchScript type system.*")
        self.temp = tempfile.TemporaryDirectory(prefix="test37-check-")
        self.root = Path(self.temp.name)
        self.mapping = {
            "condition": {"temperature_k": 300, "system": "synthetic_software_test_only"},
            "source_is_equilibrium": False,
            "species": [dict(name=name, atomic_number=z, mass_amu=mass) for name, z, mass in
                        [("clay_Al",13,26.98),("clay_Mg",12,24.30),("clay_O",8,15.999),
                         ("Na",11,22.99),("Ca",20,40.078)]],
            "sites": [dict(species="clay_Al", indices=[0,1], weights=[0.7,0.3])] +
                     [dict(species=name, indices=[i]) for name, i in
                      [("clay_Mg",2),("clay_O",3),("Na",4),("Ca",5),
                       ("clay_Al",9),("clay_O",10),("Na",11),("Ca",12)]],
        }
        (self.root/"mapping.json").write_text(json.dumps(self.mapping))
        rand = np.random.default_rng(21)
        numbers = [13,13,12,8,11,20,8,1,1,13,8,11,20]
        origin = np.array([[2+2*(i%3), 2+2*((i//3)%3), 2+2*(i//9)] for i in range(13)])
        frames = [Atoms(numbers, positions=origin+rand.normal(0,0.03,origin.shape),
                        cell=np.diag([20.,20.,20.]), pbc=True) for _ in range(6)]
        ase.io.write(self.root/"source.extxyz", frames)

    def tearDown(self):
        self.temp.cleanup()
        self.warnings.__exit__(None,None,None)

    def run_stage(self, arguments):
        args = t.parser().parse_args(arguments)
        with contextlib.redirect_stdout(io.StringIO()):
            return args.handler(args)

    def prepare(self):
        self.run_stage(["prepare", "--trajectory", str(self.root/"source.extxyz"),
                        "--mapping", str(self.root/"mapping.json"), "--output", str(self.root/"data")])

    def train(self, out, updates, resume=False):
        args = ["train", "--dataset", str(self.root/"data"), "--output", str(self.root/out),
                "--device", "cpu", "--updates", str(updates), "--batch-size", "1",
                "--checkpoint-every", "1", "--log-every", "1"]
        if resume:
            args.append("--resume")
        self.run_stage(args)

    def test_original_architecture_and_forward(self):
        # Compile just the original class/model assignment. Importing test32
        # itself would start its 3000-atom training/generation at module scope.
        source = ast.parse((t.ROOT/"test32.py").read_text())
        embed = next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=="InitialEmbedding")
        expr = next(n.value for n in source.body if isinstance(n,ast.Assign)
                    and any(isinstance(x,ast.Name) and x.id=="model" for x in n.targets))
        scope = vars(t).copy()
        exec(compile(ast.Module(body=[embed], type_ignores=[]),"test32.py","exec"),scope)
        scope.update(NUM_SPECIES=2,CUTOFF=5.0,device=torch.device("cpu"))
        scope['init_embed']=scope['InitialEmbedding'](2,5.0)
        original=eval(compile(ast.Expression(expr),"test32.py","eval"),scope).eval()
        current=t.build_model(t.architecture(2,5.0),torch.device("cpu")).eval()
        current.load_state_dict(original.state_dict())
        positions=np.array([[1,1,1],[3,1,1],[2,3,1]],dtype=float)
        a=t.graph(positions,np.eye(3)*20,[0,1,0],5,torch.device("cpu"))
        with torch.no_grad():
            torch.testing.assert_close(current(a.clone()),original(a.clone()),rtol=0,atol=0)

    def test_mapping_periodic_centroid_and_rejections(self):
        a=Atoms('Al2',positions=[[.1,0,0],[9.9,0,0]],cell=[10,10,10],pbc=True)
        mapping=dict(sites=[dict(indices=[0,1],weights=[0.5,0.5])])
        pos,_=t.map_frame(a,mapping)
        self.assertLess(min(abs(pos[0,0]), abs(pos[0,0]-10)),1e-6)
        mapping['sites'].append(dict(indices=[0]))
        with self.assertRaisesRegex(ValueError,"disjoint"):
            t.map_frame(a,mapping)
        with self.assertRaisesRegex(ValueError,"integers"):
            t.map_frame(a,dict(sites=[dict(indices=[0.5])]))

    def test_end_to_end_and_resume(self):
        self.prepare()
        p,c,meta=t.load_dataset(self.root/'data')
        self.assertEqual(p.shape,(6,9,3))
        self.assertNotIn(1,[s['atomic_number'] for s in meta['species']])
        self.assertFalse(meta['source_is_equilibrium'])
        self.train('split',2)
        self.train('split',4,resume=True)
        self.train('full',4)
        load=lambda name: torch.load(self.root/name/'checkpoint.pt',map_location='cpu',weights_only=False)
        a,b=load('split'),load('full')
        for key in a['model_state_dict']:
            torch.testing.assert_close(a['model_state_dict'][key],b['model_state_dict'][key],rtol=0,atol=0)
        ck=self.root/'full/checkpoint.pt'
        self.run_stage(['generate','--checkpoint',str(ck),'--output',str(self.root/'gen'),
                        '--device','cpu','--noisy-steps','1','--polish-steps','1'])
        model,checkpoint=t.checkpoint_model(ck,torch.device('cpu'))
        pos=torch.tensor(checkpoint['start_positions_angstrom'])
        torch.manual_seed(1337)
        with torch.no_grad():
            data=t.graph(pos.numpy(),c[0],meta['type_ids'],checkpoint['architecture']['cutoff_angstrom'],torch.device('cpu'))
            expected=pos-model(data)-torch.randn_like(pos)
        generated=np.load(self.root/'gen/positions.npy')
        np.testing.assert_allclose(generated[1],expected.numpy(),atol=1e-6)
        # Interrupted generation only advertises initialized frames; resuming
        # must restore the same noise sequence and preserve all generated frames.
        generation_args=['generate','--checkpoint',str(ck),'--output',str(self.root/'gen-resume'),
                         '--device','cpu','--noisy-steps','1','--polish-steps','1']
        t.STOP=True
        try:
            self.assertEqual(self.run_stage(generation_args),75)
        finally:
            t.STOP=False
        state=json.loads((self.root/'gen-resume/generation.json').read_text())
        self.assertEqual(state['valid_frames'],1)
        self.assertEqual(state['settings']['cutoff_angstrom'],10.0)
        self.assertEqual(state['settings']['checkpoint_sha256'],t.digest(ck))
        # Old runs used the larger graph and did not record its cutoff. Mixing
        # their saved frames with the corrected sampler must be rejected.
        restart_path=self.root/'gen-resume/generation_restart.pt'
        restart=torch.load(restart_path,map_location='cpu',weights_only=False)
        cutoff=restart['settings'].pop('cutoff_angstrom')
        t.save_checkpoint(restart_path,restart)
        with self.assertRaisesRegex(ValueError,'including graph cutoff'):
            self.run_stage(generation_args+['--resume'])
        restart['settings']['cutoff_angstrom']=cutoff
        t.save_checkpoint(restart_path,restart)
        self.run_stage(generation_args+['--resume'])
        np.testing.assert_array_equal(np.load(self.root/'gen-resume/positions.npy'),generated)
        self.run_stage(['analyze','--reference',str(self.root/'data'),'--samples',str(self.root/'gen'),
                        '--output',str(self.root/'generation-analysis')])
        self.run_stage(['md','--checkpoint',str(ck),'--output',str(self.root/'md'),
                        '--device','cpu','--steps','2','--save-every','1','--sigma-ref','0.1'])
        self.assertTrue((self.root/'md/run_metrics.json').exists())
        self.run_stage(['analyze','--reference',str(self.root/'data'),'--samples',str(self.root/'md/md.extxyz'),
                        '--output',str(self.root/'analysis')])
        self.assertTrue((self.root/'analysis/rdf.json').exists())
        self.run_stage(['export','--checkpoint',str(ck),'--output',str(self.root/'export'),'--sigma-ref','0.1'])
        bundle=torch.load(self.root/'export/cg_score_forcefield.pt',weights_only=False)
        self.assertFalse(bundle['provides_energy'])
        self.assertEqual(bundle['num_atoms'],9)
        self.assertEqual(bundle['architecture']['irreps_out'],'1x1e')
        # Reject new runs overwriting existing results and modified input data.
        with self.assertRaises(ValueError):
            self.train('full',4)
        path=self.root/'data/positions.npy'
        values=np.load(path);values[0,0,0]+=0.1;np.save(path,values)
        with self.assertRaisesRegex(ValueError,'modified'):
            t.load_dataset(self.root/'data')


if __name__ == '__main__':
    unittest.main()
