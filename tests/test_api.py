from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from smiles_service.api import app
from smiles_service.store import Store

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('SMILES_DATA_DIR',str(tmp_path));monkeypatch.delenv('SMILES_API_KEY',raising=False)
    return TestClient(app)

def test_molecule_restrictions(client):
    for s in ['not_a_smiles','CC.[Na+]','[CH3]','[Na+]','F[C@](Cl)(Br)I.C','CC(O)C(=O)O','[Fe]']:
        assert client.post('/api/jobs',json={'smiles':s}).status_code==422
    r=client.post('/api/preview',json={'smiles':'COc1ccccc1'})
    assert r.status_code==200 and r.json()['formula']=='C7H8O'
    assert '<svg' in r.json()['svg']

def test_queue_claim_and_download_gating(client):
    r=client.post('/api/jobs',json={'smiles':'CCCC'});assert r.status_code==202
    job=r.json()['id'];store=Store()
    assert client.get(f'/api/jobs/{job}/downloads/all').status_code==409
    assert store.claim()['id']==job
    assert store.claim() is None
    store.update(job,status='failed',stage='Stopped',error='Validation failed')
    assert client.get(f'/api/jobs/{job}').json()['status']=='failed'
    assert client.get(f'/api/jobs/{job}/downloads/all').status_code==409

def test_key_and_path_checks(client,monkeypatch):
    monkeypatch.setenv('SMILES_API_KEY','test-key')
    assert client.post('/api/jobs',json={'smiles':'CCCC'}).status_code==401
    r=client.post('/api/jobs',json={'smiles':'CCCC'},headers={'X-API-Key':'test-key'})
    assert r.status_code==202
    assert client.get('/api/jobs/not-an-id',headers={'X-API-Key':'test-key'}).status_code==404
    assert client.get('/api/jobs/'+r.json()['id']).status_code==401

def test_claim_is_atomic(tmp_path):
    store=Store(tmp_path);jobs=[store.create('CCCC')['id'] for _ in range(10)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed=list(pool.map(lambda _:Store(tmp_path).claim(),range(12)))
    assert sorted(r['id'] for r in claimed if r)==sorted(jobs)

def test_queue_limit_and_expiry(client,monkeypatch):
    monkeypatch.setenv('SMILES_MAX_PENDING','1')
    r=client.post('/api/jobs',json={'smiles':'CCCC'}).json()
    assert client.post('/api/jobs',json={'smiles':'CC'}).status_code==429
    store=Store();store.update(r['id'],status='succeeded',result={})
    folder=store.root/r['id'];folder.mkdir();(folder/'all.zip').write_bytes(b'test-artifact')
    assert client.get(f"/api/jobs/{r['id']}/downloads/all").status_code==200
    store.expire(ttl=-1)
    assert client.get(f"/api/jobs/{r['id']}/downloads/all").status_code==410
    assert not folder.exists()
