const $=id=>document.getElementById(id);
const base=()=>($('endpoint').value.trim()||'').replace(/\/$/,'');
const headers=()=>({'Content-Type':'application/json',...($('key').value?{'X-API-Key':$('key').value}:{})});
let pollTimer=null,previewURL=null,submissionBase='';
const message=(text,error=false)=>{$('message').textContent=text;$('message').className=error?'error':'';};
async function request(path,options={}){
 if(location.hostname.endsWith('github.io')&&!base())throw new Error('Set your chemistry API URL in Connection settings. This website is hosted separately from the simulation server.');
 const response=await fetch(base()+path,{...options,headers:headers()});
 if(!response.ok){const body=await response.json().catch(()=>({}));throw new Error(typeof body.detail==='string'?body.detail:`Server returned ${response.status}.`);}
 return response;
}
function connectionHint(error){return error instanceof TypeError?'Cannot reach the API. Start the backend or enter its HTTPS URL in Connection settings.':error.message;}
async function preview(){
 try{message('Checking structure…');
 const data=await(await request('/api/preview',{method:'POST',body:JSON.stringify({smiles:$('smiles').value})})).json();
 if(previewURL)URL.revokeObjectURL(previewURL);
 previewURL=URL.createObjectURL(new Blob([data.svg],{type:'image/svg+xml'}));
 $('molecule').src=previewURL;$('molecule').style.display='block';
 $('identity').textContent=`${data.formula} · ${data.heavy_atoms} heavy atoms · charge ${data.formal_charge}`;
 message('Structure checked. Parameters are assigned to this molecular state.');return data;
 }catch(error){message(connectionHint(error),true);return null;}
}
$('preview').addEventListener('click',preview);
document.querySelectorAll('[data-smiles]').forEach(button=>button.addEventListener('click',()=>{$('smiles').value=button.dataset.smiles;preview();}));
$('endpoint').value=new URLSearchParams(location.search).get('api')||'';
$('endpoint').addEventListener('change',()=>{$('docs').href=base()+'/docs';});
$('docs').href=base()+'/docs';
$('generate').addEventListener('submit',async event=>{
 event.preventDefault();if(pollTimer)clearTimeout(pollTimer);
 $('submit').disabled=true;
 try{
  const info=await preview();if(!info)return;
  const data=await(await request('/api/jobs',{method:'POST',body:JSON.stringify({smiles:info.canonical_smiles})})).json();
  submissionBase=base();$('results').hidden=false;$('downloads').replaceChildren();$('validation').replaceChildren();
  $('jobid').textContent='JOB '+data.id;$('jobstage').textContent='Waiting for worker';$('progress').className='progress';
  $('results').scrollIntoView({behavior:'smooth',block:'start'});message('Job submitted. Keep this page open to receive your downloads.');
  poll(data.id);
 }catch(error){message(connectionHint(error),true);}
 finally{$('submit').disabled=false;}
});
async function poll(id){
 try{
  if(base()!==submissionBase)throw new Error('The API address changed. Restore the original address to check this job.');
  const job=await(await request('/api/jobs/'+id)).json();$('jobstage').textContent=job.stage;
  if(job.status==='succeeded'){
   $('progress').className='progress done';message('Packages are ready. Review validation scope before use.');
   for(const [kind,path] of Object.entries(job.downloads)){
    const button=document.createElement('button');button.textContent=(kind==='all'?'All packages':kind.toUpperCase())+' ↓ ZIP';
    button.addEventListener('click',async()=>{button.disabled=true;try{
     const response=await request(path);const blob=await response.blob();const url=URL.createObjectURL(blob);
     const a=document.createElement('a');a.href=url;a.download=`forcefield-${kind}-${id.slice(0,8)}.zip`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }catch(e){message(connectionHint(e),true);}finally{button.disabled=false;}});$('downloads').append(button);
   }
   renderValidation(job.result);return;
  }
  if(['failed','expired'].includes(job.status)){$('progress').hidden=true;message(job.error||'Downloads expired.',true);return;}
  pollTimer=setTimeout(()=>poll(id),2000);
 }catch(error){message(connectionHint(error)+' Job ID: '+id,true);}
}
function renderValidation(result){
 const box=document.createElement('div');box.className='checks';const title=document.createElement('strong');title.textContent=result.forcefield+' · '+result.charges;box.append(title);
 const p=document.createElement('p');p.textContent=result.scope;box.append(p);
 const table=document.createElement('table');const head=table.createTHead().insertRow();
 for(const text of ['Check','Energy error (kJ/mol)','Max force error (kJ/mol/nm)']){const th=document.createElement('th');th.textContent=text;head.append(th);}
 for(const r of result.validation.results){const row=table.insertRow();for(const text of [r.route,Number(r.energy_abs_error).toPrecision(3),Number(r.force_max_abs_error).toPrecision(3)]){const td=row.insertCell();td.textContent=text;}}
 box.append(table);$('validation').append(box);
}

if(location.hostname.endsWith('github.io')&&!base())message('Website ready. Set the chemistry API URL in Connection settings to generate packages.');
