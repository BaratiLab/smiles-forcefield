"""Transport test ONLY: deterministic first-ID selection, not an LLM."""
import json
import sys
s = json.load(sys.stdin)
print(json.dumps({'candidate_ids':[r['id'] for r in s['candidates'][:s['batch_size']]],
                  'hypothesis':'MOCK transport check; no scientific reasoning'}))
