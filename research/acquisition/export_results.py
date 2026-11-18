"""Bundle only a fixed five-molecule experiment, never production job data."""
import argparse,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',default='aws-runs-compact');a=p.parse_args()
root=Path(__file__).resolve().parent
runs=root/a.runs
if runs.parent!=root or not (runs/'status.json').is_file():raise ValueError('Expected a completed/failed suite directory directly under research root')
status=json.loads((runs/'status.json').read_text())
required={(m,p) for m in ['butane','biphenyl','anisole','nma','phenylethanol'] for p in ['random','bayes','llm']}
if not required<={(r['molecule'],r['policy']) for r in status}:raise ValueError('Suite has not finished all five molecules and three policies')
out=root/'aws-acquisition-results.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for folder in [root/'aws-data',runs]:
  for item in sorted(folder.rglob('*')):
   if item.is_file() and '__pycache__' not in str(item):z.write(item,str(item.relative_to(root)))
 for name in ['aws-environment.json','README.md']:
  if (root/name).is_file():z.write(root/name,name)
 z.writestr('SCOPE.txt','MMFF94 minus UFF proxy; NOT quantum chemistry. Qwen2.5:0.5b private CPU acquisition, compact-v2 prompts. Fixed development molecules. See status.json for failed runs. One seed, no superiority claim.\n')
print(out)
