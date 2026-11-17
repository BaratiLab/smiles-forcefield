import hmac
import os
from pathlib import Path
import re
import shutil
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .store import Store
from .molecules import inspect, svg

class Submission(BaseModel):
    smiles:str=Field(min_length=1,max_length=1000)

app=FastAPI(title='SMILES Forcefield',version='0.1.0',description='Sage 2.2.1 / AM1-BCC. Isolated-molecule export consistency checks. Native NAMD is not tested.')
origins=[x.strip() for x in os.getenv('SMILES_CORS_ORIGINS','').split(',') if x.strip()]
if origins:app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type','X-API-Key'])

def authorize(key):
    configured=os.getenv('SMILES_API_KEY')
    if configured and not hmac.compare_digest((key or '').encode(),configured.encode()):raise HTTPException(401,'Invalid API key.')

def job_or_404(job):
    if not re.fullmatch('[0-9a-f]{32}',job):raise HTTPException(404,'Job not found.')
    record=Store().get(job)
    if not record:raise HTTPException(404,'Job not found.')
    return record

@app.get('/api/health')
def health():
    return {'status':'ok','gromacs_available':bool(shutil.which('gmx')),'lammps_available':bool(shutil.which('lmp')),'namd_native_validation':False}

@app.post('/api/preview')
def preview(body:Submission):
    try:info=inspect(body.smiles)
    except ValueError as e:raise HTTPException(422,str(e))
    return {**info,'svg':svg(info['canonical_smiles'])}

@app.post('/api/jobs',status_code=202)
def submit(body:Submission,x_api_key:str|None=Header(default=None)):
    authorize(x_api_key)
    try:
        info=inspect(body.smiles)
        job=Store().create(info['canonical_smiles'],int(os.getenv('SMILES_MAX_PENDING','20')))
    except ValueError as e:raise HTTPException(422,str(e))
    except OverflowError as e:raise HTTPException(429,str(e))
    return {'id':job['id'],'status':job['status'],'status_url':f"/api/jobs/{job['id']}",'molecule':info}

@app.get('/api/jobs/{job}')
def status(job:str,x_api_key:str|None=Header(default=None)):
    authorize(x_api_key)
    r=job_or_404(job)
    result=r.get('result')
    return {'id':r['id'],'smiles':r['smiles'],'status':r['status'],'stage':r['stage'],
            'created':r['created'],'updated':r['updated'],'error':r['error'],'result':result,
            'downloads':{name:f'/api/jobs/{job}/downloads/{name}' for name in ['all','gromacs','lammps','namd']} if r['status']=='succeeded' else {}}

@app.get('/api/jobs/{job}/downloads/{engine}')
def download(job:str,engine:str,x_api_key:str|None=Header(default=None)):
    authorize(x_api_key)
    r=job_or_404(job)
    if engine not in {'all','gromacs','lammps','namd'}:raise HTTPException(404,'Unknown package.')
    if r['status']=='expired':raise HTTPException(410,'Downloads expired.')
    if r['status']!='succeeded':raise HTTPException(409,'Validated packages are not ready.')
    file=Store().root/job/(engine+'.zip')
    if not file.is_file():raise HTTPException(410,'Download unavailable.')
    return FileResponse(file,media_type='application/zip',filename=f'smiles-forcefield-{engine}-{job[:8]}.zip',headers={'Cache-Control':'no-store'})

static=Path(__file__).parent/'static'
if static.exists():app.mount('/',StaticFiles(directory=static,html=True),name='web')
