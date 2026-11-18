"""Budget-matched torsion acquisition benchmark. Python >=3.9, NumPy only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parent

def features(angles, harmonics=3):
    """Angles in radians. Intercept absorbs arbitrary molecular energy zero."""
    a = np.asarray(angles, dtype=float)
    cols = [np.ones(len(a))]
    for j in range(a.shape[1]):
        for n in range(1, harmonics + 1):
            cols.extend([np.cos(n*a[:, j]), np.sin(n*a[:, j])])
    return np.column_stack(cols)

class Ridge:
    """Gaussian linear posterior; fixed prior/noise shared by all policies."""
    def __init__(self, x, y, alpha=1., noise=.1):
        self.precision = alpha*np.eye(x.shape[1]) + x.T@x/noise**2
        self.cov = np.linalg.solve(self.precision, np.eye(x.shape[1]))
        self.mean = self.cov@x.T@y/noise**2
        self.noise = noise
    def predict(self, x):
        return x@self.mean
    def variance(self, x):
        return np.einsum('ij,jk,ik->i', x, self.cov, x)

def bayes_batch(model, x, available, batch):
    """Greedy conditional information gain; covariance updated within batch."""
    cov = model.cov.copy()
    remaining = list(available)
    selected = []
    for _ in range(batch):
        scores = np.einsum('ij,jk,ik->i', x[remaining], cov, x[remaining])
        pos = int(np.argmax(scores))
        idx = remaining.pop(pos)
        selected.append(idx)
        v = cov@x[idx]
        cov -= np.outer(v, v)/(model.noise**2 + x[idx]@v)
    return selected

def validate_response(response, allowed, batch):
    ids = response.get('candidate_ids')
    if not isinstance(ids, list) or len(ids) != batch:
        raise ValueError('LLM must return exactly batch_size candidate IDs')
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != batch:
        raise ValueError('LLM IDs must be unique strings')
    if not set(ids) <= set(allowed):
        raise ValueError('LLM selected unavailable or unknown IDs')
    return ids

def llm_batch(command, molecule, pool, selected, available, model, x, batch, logdir, step, metadata=None):
    """External trusted adapter: JSON on stdin, JSON on stdout; no shell."""
    state = {
        'dataset_metadata': metadata,
        'target_kind': 'reference energy minus baseline energy; reference may be MM proxy, consult dataset metadata',
        'instruction': 'Select informative configurations to fit an additive Fourier torsion correction. Return JSON {"candidate_ids": [...], "hypothesis": "..."}. Do not invent labels. All angles are radians and energies kcal/mol.',
        'molecule': molecule, 'batch_size': batch, 'fit': {'basis':'additive cosine/sine Fourier', 'noise_kcal_mol':model.noise},
        'observations': [dict(id=pool[i]['id'], angles_rad=pool[i]['angles_rad'],
                              correction_kcal_mol=pool[i]['qm_energy_kcal_mol']-pool[i]['baseline_energy_kcal_mol'],
                              fitted_correction_kcal_mol=float(model.predict(x[i:i+1])[0])) for i in selected],
        'candidates': [dict(id=pool[i]['id'], angles_rad=pool[i]['angles_rad'],
                            predicted_correction_kcal_mol=float(model.predict(x[i:i+1])[0]),
                            variance=float(model.variance(x[i:i+1])[0])) for i in available]
    }
    request = json.dumps(state)
    (logdir/f'{step:03d}_request.json').write_text(request)
    started = time.monotonic()
    result = subprocess.run(command, input=request, text=True, capture_output=True, timeout=1300, check=True)
    (logdir/f'{step:03d}_response.json').write_text(result.stdout)
    response = json.loads(result.stdout)
    ids = validate_response(response, [pool[i]['id'] for i in available], batch)
    lookup = {pool[i]['id']: i for i in available}
    return [lookup[i] for i in ids], time.monotonic()-started

def load_dataset(path):
    d = json.loads(Path(path).read_text())
    if d.get('units') != {'angles':'radian', 'energy':'kcal/mol'}:
        raise ValueError('Explicit radian and kcal/mol units required')
    nt = len(d['molecule']['torsions'])
    seen = set()
    geometry_seen = set()
    for split in ('pool', 'test'):
        if not d[split]:
            raise ValueError('Empty split')
        for r in d[split]:
            if r['id'] in seen:
                raise ValueError('Duplicate ID across or within splits')
            seen.add(r['id'])
            if r.get('geometry_sha256'):
                if r['geometry_sha256'] in geometry_seen:
                    raise ValueError('Duplicate geometry across or within splits')
                geometry_seen.add(r['geometry_sha256'])
            if len(r['angles_rad']) != nt:
                raise ValueError('Incorrect torsion dimension')
            values = r['angles_rad']+[r['qm_energy_kcal_mol'], r['baseline_energy_kcal_mol']]
            if not np.isfinite(values).all():
                raise ValueError('Non-finite data')
    if d['metadata'].get('kind') not in {'synthetic','mm_proxy'}:
        if not d['metadata'].get('qm_protocol') or not d['metadata'].get('baseline_protocol'):
            raise ValueError('Real datasets require QM and baseline provenance')
    return d

def run(args):
    d = load_dataset(args.data)
    pool, test = d['pool'], d['test']
    if not 1 <= args.initial <= args.budget <= len(pool) or args.batch < 1:
        raise ValueError('Require 1 <= initial <= budget <= pool size and positive batch')
    if args.alpha <= 0 or args.noise <= 0:
        raise ValueError('Prior precision and noise must be positive')
    if 'llm' in args.policies and not args.llm_command:
        raise ValueError('LLM requested: provide --llm-command JSON argv; no silent fallback')
    command = json.loads(args.llm_command) if args.llm_command else None
    if command is not None and (not isinstance(command, list) or not command or any(not isinstance(s,str) for s in command)):
        raise ValueError('Command must be a nonempty JSON array of strings')
    x = features([r['angles_rad'] for r in pool], args.harmonics)
    xt = features([r['angles_rad'] for r in test], args.harmonics)
    y = np.array([r['qm_energy_kcal_mol']-r['baseline_energy_kcal_mol'] for r in pool])
    yt = np.array([r['qm_energy_kcal_mol']-r['baseline_energy_kcal_mol'] for r in test])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict({k:v for k,v in vars(args).items() if k != 'func'}, dataset_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    dataset_kind=d['metadata']['kind'], controller_kind=getattr(args,'controller_kind','unspecified'), numpy_version=np.__version__)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    baseline = float(np.sqrt(np.mean((yt-yt.mean())**2)))
    oracle = Ridge(x, y, args.alpha, args.noise)
    fullerr = oracle.predict(xt)-yt
    fullrmse = float(np.sqrt(np.mean((fullerr-fullerr.mean())**2)))
    for seed in args.seeds:
        initial = np.random.default_rng(seed).choice(len(pool), args.initial, replace=False).tolist()
        for policy in args.policies:
            rng = np.random.default_rng(seed+100000)
            selected = initial.copy()
            logdir = out/f'{policy}_{seed}'
            logdir.mkdir()
            latency = 0.
            step = 0
            while True:
                model = Ridge(x[selected], y[selected], args.alpha, args.noise)
                # Evaluator only: metrics are never passed back to acquisition policy.
                err = model.predict(xt)-yt
                rmse = float(np.sqrt(np.mean((err-err.mean())**2)))
                rows.append(dict(molecule=d['molecule']['id'], policy=policy, seed=seed,
                                 queries=len(selected), relative_energy_rmse=rmse,
                                 baseline_rmse=baseline, full_pool_rmse=fullrmse,
                                 llm_seconds=latency))
                (logdir/'selected_ids.json').write_text(json.dumps([pool[i]['id'] for i in selected]))
                np.save(logdir/'coefficients.npy', model.mean)
                if len(selected) == args.budget:
                    break
                available = sorted(set(range(len(pool)))-set(selected))
                batch = min(args.batch, args.budget-len(selected))
                if policy == 'random':
                    chosen = rng.choice(available, batch, replace=False).tolist()
                elif policy == 'bayes':
                    chosen = bayes_batch(model, x, available, batch)
                else:
                    chosen, elapsed = llm_batch(command, d['molecule'], pool, selected, available,
                                                model, x, batch, logdir, step, d['metadata'])
                    latency += elapsed
                (logdir/f'{step:03d}_acquired.json').write_text(json.dumps([pool[i]['id'] for i in chosen]))
                selected.extend(chosen)
                step += 1
    with (out/'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = []
    for policy in args.policies:
        curves = [[r for r in rows if r['policy']==policy and r['seed']==s] for s in args.seeds]
        aucs = [sum((b['queries']-a['queries'])*(a['relative_energy_rmse']+b['relative_energy_rmse'])/2
                    for a,b in zip(c,c[1:])) for c in curves]
        summary.append(dict(policy=policy, mean_final_rmse=float(np.mean([c[-1]['relative_energy_rmse'] for c in curves])),
                            mean_error_query_auc=float(np.mean(aucs))))
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(dataset_kind=d['metadata']['kind'], results=summary), indent=2))

def synthetic(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for k,m in enumerate(json.loads((ROOT/'molecules.json').read_text())):
        rng = np.random.default_rng(440+k)
        nt = len(m['torsions'])
        coeff = rng.normal(0, .7, 1+nt*6)
        coeff[0] = 2.
        d = dict(molecule=m, units=dict(angles='radian', energy='kcal/mol'),
                 metadata=dict(kind='synthetic', warning='Not QM. Toy arbitrary coefficients, not molecule-specific physics.'))
        for split,n in [('pool',160), ('test',120)]:
            a = rng.uniform(-np.pi,np.pi,(n,nt)); e = features(a)@coeff
            if nt==2:
                e += .5*np.cos(a[:,0]-a[:,1])  # intentional unrepresentable coupling
            e += rng.normal(0,.02,n)
            d[split] = [dict(id=f'{split}_{i:04d}', angles_rad=a[i].tolist(),
                            baseline_energy_kcal_mol=0., qm_energy_kcal_mol=float(e[i])) for i in range(n)]
        (out/f'{m["id"]}.json').write_text(json.dumps(d, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)
    s = sub.add_parser('synthetic'); s.add_argument('--out', required=True); s.set_defaults(func=synthetic)
    r = sub.add_parser('run')
    r.add_argument('--data', required=True); r.add_argument('--out', required=True)
    r.add_argument('--policies', nargs='+', choices=['random','bayes','llm'], default=['random','bayes'])
    r.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    r.add_argument('--initial', type=int, default=8); r.add_argument('--budget', type=int, default=64)
    r.add_argument('--batch', type=int, default=8); r.add_argument('--harmonics', type=int, default=3)
    r.add_argument('--alpha', type=float, default=1.); r.add_argument('--noise', type=float, default=.1)
    r.add_argument('--controller-kind', choices=['ollama_llm','mock','external_llm'], default='external_llm')
    r.add_argument('--llm-command', help='JSON argv for trusted external adapter')
    r.set_defaults(func=run)
    args = parser.parse_args(); args.func(args)
