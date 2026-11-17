"""Run separately from API: python -m smiles_service.worker [--once]."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from .store import Store

def process(store,job):
    folder=store.root/job['id'];folder.mkdir(exist_ok=True)
    (folder/'request.json').write_text(json.dumps({'smiles':job['smiles']}))
    timeout=int(os.getenv('SMILES_JOB_TIMEOUT','900'))
    started=time.monotonic()
    with (folder/'worker.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'-m','smiles_service.pipeline',str(folder)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                progress=folder/'progress.json'
                if progress.exists():
                    try:store.update(job['id'],stage=json.loads(progress.read_text())['stage'])
                    except (json.JSONDecodeError,KeyError):pass
                if time.monotonic()-started>timeout:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise TimeoutError('Job exceeded execution time limit.')
                time.sleep(.5)
            if p.returncode:raise RuntimeError('Generation or validation failed. Check the worker log on the server; no downloads released.')
            report=json.loads((folder/'result.json').read_text())
            store.update(job['id'],status='succeeded',stage='Packages ready',result=report)
        except Exception as e:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
            store.update(job['id'],status='failed',stage='Stopped',error=str(e))

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    store=Store()
    # A killed worker's jobs are never silently retried (avoids overlapping computations).
    # Operators can inspect these records and resubmit; queue remains durable.
    while True:
        store.expire(int(os.getenv('SMILES_RETENTION_SECONDS','86400')))
        job=store.claim()
        if job:
            try:process(store,job)
            except Exception as e:store.update(job['id'],status='failed',stage='Worker error',error=type(e).__name__+': could not execute the job.')
        elif a.once:return
        else:time.sleep(1)
        if a.once:return
if __name__=='__main__':main()
