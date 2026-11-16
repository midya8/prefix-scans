"""Full bivariate polynomial references for dense squared-loss curvature."""
import copy
from fractions import Fraction as F
import random
import unittest
from scanlab import curvature_audit
from scanlab.exact import jsonify
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
from scanlab.dense import _dot
from test_dense import flat, layout, case
from test_adjoint import add, mul


def polynomial(inp, first, second):
    n, d = len(inp['pairs']), len(inp['initial'])
    lifted = layout([{(0,0): F(x), (1,0): F(v), (0,1): F(w)}
                     for x,v,w in zip(flat(inp),flat(first),flat(second))], n,d)
    h = lifted['initial']; states=[]; loss={}
    for t, step in enumerate(lifted['pairs']):
        new=[]
        for i in range(d):
            value=step['write'][i]
            for j in range(d):value=add(value,mul(step['matrix'][i][j],h[j]))
            new.append(value)
        h=new;states.append(h)
        for i in range(d):
            residual=add(h[i],{(0,0):-F(inp['targets'][t][i])})
            loss=add(loss,{k:v/2 for k,v in mul(residual,residual).items()})
    return states,loss


def model(n,d,seed):
    inp=case(n,d,seed);inp.pop('cotangents')
    rng=random.Random(seed+982)
    inp['targets']=[[F(rng.randint(-2,2),3) for _ in range(d)]for _ in range(n)]
    return inp


class DenseCurvature(unittest.TestCase):
    def test_all_672_coordinates_against_full_bivariate_polynomials(self):
        checked=0
        for n in range(6):
            for d in (1,2,3):
                for seed in range(2):
                    inp=model(n,d,100*n+10*d+seed);r=curvature_audit(**inp);p=len(flat(inp))
                    for k in range(p):
                        basis=layout([F(i==k)for i in range(p)],n,d)
                        _,loss=polynomial(inp,inp['direction'],basis)
                        self.assertEqual(flat(r['gradient'])[k],loss.get((0,1),0))
                        self.assertEqual(flat(r['hessian_vector'])[k],loss.get((1,1),0));checked+=1
        self.assertEqual(checked,672)

    def test_24_full_directional_models(self):
        for n in range(8):
            for d in (1,2,3):
                inp=model(n,d,19*n+d);r=curvature_audit(**inp);zero=layout([F(0)]*len(flat(inp)),n,d)
                states,loss=polynomial(inp,inp['direction'],zero)
                self.assertEqual(r['states'],[tuple(x.get((0,0),0)for x in row)for row in states])
                self.assertEqual(r['state_jvp'],[tuple(x.get((1,0),0)for x in row)for row in states])
                self.assertEqual(r['state_second_directional'],[tuple(2*x.get((2,0),0)for x in row)for row in states])
                self.assertEqual(r['squared_loss'],loss.get((0,0),0));self.assertEqual(r['directional_first'],loss.get((1,0),0));self.assertEqual(r['directional_second'],2*loss.get((2,0),0))
                self.assertEqual(r['witness']['first_identity_gap'],0);self.assertEqual(r['witness']['second_identity_gap'],0)

    def test_24_hessian_symmetries(self):
        for n in range(8):
            for d in (1,2,3):
                inp=model(n,d,7*n+d);other=model(n,d,1000+n*d)['direction'];r=curvature_audit(**inp);rr=curvature_audit(**dict(inp,direction=other))
                self.assertEqual(_dot(r['hessian_vector'],other),_dot(rr['hessian_vector'],inp['direction']))

    def test_residual_curvature_can_be_negative(self):
        inp={'pairs':[{'matrix':[[0]],'write':[0]}],'initial':[0],
             'direction':{'pairs':[{'matrix':[[1]],'write':[0]}],'initial':[1]},'targets':[[1]]}
        r=curvature_audit(**inp)
        self.assertEqual(r['squared_loss'],F(1,2));self.assertEqual(r['state_jvp'],[(F(0),)])
        self.assertEqual(r['state_second_directional'],[(F(2),)]);self.assertEqual(r['directional_second'],-2)
        self.assertEqual(r['hessian_vector']['initial'],(F(-1),));self.assertEqual(r['hessian_vector']['pairs'][0]['matrix'],((F(-1),),))
        # Gauss-Newton keeps only squared state tangents and misses this curvature.
        self.assertEqual(sum(v*v for row in r['state_jvp']for v in row),0)

    def test_empty_singular_maximum_and_zero_direction(self):
        for n,d in ((0,4),(1,1),(32,4)):
            inp={'pairs':[{'matrix':[[0]*d for _ in range(d)],'write':[1]*d}for _ in range(n)],'initial':[0]*d,
                 'direction':{'pairs':[{'matrix':[[0]*d for _ in range(d)],'write':[0]*d}for _ in range(n)],'initial':[0]*d},'targets':[[0]*d for _ in range(n)]}
            r=curvature_audit(**inp);self.assertEqual(r['directional_first'],0);self.assertEqual(r['directional_second'],0);self.assertTrue(all(v==0 for v in flat(r['hessian_vector'])))

    def test_isolation_and_validation(self):
        inp=model(3,2,17);old=copy.deepcopy(inp);state=random.getstate();r=curvature_audit(**inp)
        self.assertEqual(inp,old);self.assertEqual(random.getstate(),state);r['direction']['pairs'][0]['matrix']=None;self.assertEqual(inp,old)
        for change in ({'targets':[]},{'targets':[[True,0]]*3},{'direction':{}},{'initial':[0]*5},{'pairs':[{'matrix':[[1]],'write':[0]}]*33}):
            with self.assertRaises(ValueError):curvature_audit(**dict(inp,**change))

    def test_cli_complete_output_and_invalid_inputs(self):
        inp=model(3,2,19)
        with tempfile.TemporaryDirectory() as directory:
            f=Path(directory)/'input.json';f.write_text(json.dumps(jsonify(inp)))
            def run():return subprocess.run([sys.executable,'-m','scanlab','curvature',str(f)],cwd=ROOT,capture_output=True)
            p=run();self.assertEqual(p.returncode,0);self.assertEqual(json.loads(p.stdout),jsonify(curvature_audit(**inp)));self.assertEqual(p.stderr,b'')
            for raw in (b'{}',b'{"pairs":[],"pairs":[]}',b'NaN',b'\xff',b' '*1000001,json.dumps(dict(jsonify(inp),extra=0)).encode(),json.dumps(dict(jsonify(inp),targets=[])).encode()):
                f.write_bytes(raw);p=run();self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,b'');self.assertIn(b'error:',p.stderr);self.assertNotIn(b'Traceback',p.stderr)
