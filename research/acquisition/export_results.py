"""Bundle only the fixed five-molecule experiment, never production job data."""
import json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'aws-acquisition-results.zip'
required=root/'aws-runs/status.json'
status=json.loads(required.read_text())
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['aws-data','aws-runs']:
  for p in sorted((root/folder).rglob('*')):
   if p.is_file() and '__pycache__' not in str(p):z.write(p,str(p.relative_to(root)))
 z.writestr('SCOPE.txt','MMFF94 minus UFF proxy; NOT quantum chemistry. Qwen2.5:1.5b private CPU acquisition. Fixed development molecules. See status.json for failed runs. No superiority claim.\n')
print(out)
