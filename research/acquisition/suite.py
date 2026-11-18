"""Matched five-molecule replay; incomplete controllers remain explicit failures."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--budget',type=int,default=12);p.add_argument('--initial',type=int,default=4);p.add_argument('--batch',type=int,default=4);p.add_argument('--llm',action='store_true');a=p.parse_args()
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);status=[]
 for m in json.loads((ROOT/'molecules.json').read_text()):
  for policy in ['random','bayes']+(['llm'] if a.llm else []):
   target=out/m['id']/policy
   cmd=[sys.executable,str(ROOT/'poc.py'),'run','--data',str(Path(a.data).resolve()/(m['id']+'.json')),'--out',str(target),'--policies',policy,'--initial',str(a.initial),'--budget',str(a.budget),'--batch',str(a.batch),'--seeds',*map(str,a.seeds)]
   if policy=='llm':cmd+=['--controller-kind','ollama_llm','--llm-command',json.dumps([sys.executable,str(ROOT/'ollama_adapter.py')])]
   r=subprocess.run(cmd,text=True,capture_output=True)
   target.mkdir(parents=True,exist_ok=True);(target/'runner.log').write_text(r.stdout+r.stderr)
   status.append({'molecule':m['id'],'policy':policy,'status':'completed' if r.returncode==0 else 'failed','exit_code':r.returncode})
   (out/'status.json').write_text(json.dumps(status,indent=2));print(status[-1],flush=True)
 summarize(out)
 if any(r['status']=='failed' for r in status):raise SystemExit(1)

def summarize(out):
 out=Path(out);rows=[]
 for path in sorted(out.glob('*/*/metrics.csv')):
  with path.open() as f:rows.extend(csv.DictReader(f))
 if not rows:return
 with (out/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 summaries=[]
 for m,policy in sorted({(r['molecule'],r['policy']) for r in rows}):
  curves=[[r for r in rows if r['molecule']==m and r['policy']==policy and r['seed']==seed] for seed in sorted({r['seed'] for r in rows if r['molecule']==m and r['policy']==policy})]
  final=[float(c[-1]['relative_energy_rmse']) for c in curves]
  auc=[sum((int(b['queries'])-int(a['queries']))*(float(a['relative_energy_rmse'])+float(b['relative_energy_rmse']))/2 for a,b in zip(c,c[1:]))/(int(c[-1]['queries'])-int(c[0]['queries'])) for c in curves]
  summaries.append({'molecule':m,'policy':policy,'seeds':len(curves),'final_rmse_mean':float(np.mean(final)),'final_rmse_sd':float(np.std(final,ddof=1)) if len(final)>1 else None,'normalized_error_auc':float(np.mean(auc))})
 (out/'summary.json').write_text(json.dumps(summaries,indent=2))
 try:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,5,figsize=(17,3.6))
  for ax,m in zip(axes,sorted({r['molecule'] for r in rows})):
   for policy in sorted({r['policy'] for r in rows}):
    selected=[r for r in rows if r['molecule']==m and r['policy']==policy]
    if not selected:continue
    qs=sorted({int(r['queries']) for r in selected});vals=[[float(r['relative_energy_rmse']) for r in selected if int(r['queries'])==q] for q in qs]
    ax.plot(qs,[np.mean(v) for v in vals],marker='o',label=policy)
   ax.set_title(m);ax.set_xlabel('Queried reference energies');ax.set_ylabel('Centered RMSE (kcal/mol)');ax.legend()
  fig.suptitle('MMFF94 − UFF proxy benchmark — NOT quantum chemistry');fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=160);plt.close(fig)
 except ImportError:pass
if __name__=='__main__':main()
