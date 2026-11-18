"""Provider-neutral gateway adapter. POST benchmark state, receive action JSON.
Gateway contract is deliberately not a vendor-specific chat API.
Set LLM_GATEWAY_URL and optionally LLM_GATEWAY_TOKEN in the environment.
"""
import json
import os
import sys
import urllib.request

state = json.load(sys.stdin)
headers = {'Content-Type': 'application/json'}
if os.environ.get('LLM_GATEWAY_TOKEN'):
    headers['Authorization'] = 'Bearer ' + os.environ['LLM_GATEWAY_TOKEN']
request = urllib.request.Request(os.environ['LLM_GATEWAY_URL'],
                                 data=json.dumps(state).encode(), headers=headers, method='POST')
with urllib.request.urlopen(request, timeout=150) as response:
    action = json.load(response)
print(json.dumps(action))
