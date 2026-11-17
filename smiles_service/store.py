"""SQLite durable queue. Workers atomically claim jobs; clients poll by capability ID."""
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

class Store:
    def __init__(self, root=None):
        self.root=Path(root or os.getenv('SMILES_DATA_DIR','data')).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=self.root/'jobs.sqlite'
        with self.connect() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, smiles TEXT NOT NULL, status TEXT NOT NULL,
            stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
            error TEXT, result TEXT)''')
    def connect(self):
        c=sqlite3.connect(self.db,timeout=30)
        c.row_factory=sqlite3.Row
        return c
    def create(self,smiles,max_pending=20):
        job=uuid.uuid4().hex; now=time.time()
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            n=c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if n>=max_pending:raise OverflowError('Queue is full; try again later.')
            c.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',(job,smiles,'queued','Waiting for worker',now,now,None,None))
        return self.get(job)
    def get(self,job):
        with self.connect() as c:r=c.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone()
        if not r:return None
        d=dict(r)
        if d['result']:d['result']=json.loads(d['result'])
        return d
    def claim(self):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            r=c.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if r is None:return None
            c.execute("UPDATE jobs SET status='running',stage='Preparing molecule',updated=? WHERE id=?",(time.time(),r['id']))
        return self.get(r['id'])
    def update(self,job,**fields):
        allowed={'status','stage','error','result'}
        if not fields or not set(fields)<=allowed:raise ValueError('Invalid update')
        if 'result' in fields:fields['result']=json.dumps(fields['result'])
        fields['updated']=time.time()
        with self.connect() as c:
            c.execute('UPDATE jobs SET '+','.join(f'{k}=?' for k in fields)+' WHERE id=?',(*fields.values(),job))
    def expire(self,ttl=86400):
        """Expire terminal artifacts only; queued/running jobs are preserved."""
        import shutil
        with self.connect() as c:
            rows=c.execute("SELECT id FROM jobs WHERE status IN ('succeeded','failed') AND updated<?",(time.time()-ttl,)).fetchall()
            for r in rows:
                shutil.rmtree(self.root/r['id'],ignore_errors=True)
                c.execute("UPDATE jobs SET status='expired',stage='Downloads expired',result=NULL WHERE id=?",(r['id'],))
