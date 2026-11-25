"""Fetch three test inputs per degradation; no object labels are invented.

Uses public sources linked by the dataset/model authors. Remote GoPro/SOTS
archives are sampled via HTTP Range. RainTrainL is never downloaded.
Run with Python + requests, numpy, Pillow, gdown. Re-running preserves fixed
selection (first 3 unique scenes/sequences, lexicographic order).
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile
from collections import Counter
import numpy as np
import requests
from PIL import Image
import gdown
from remote_zip import RemoteZipReader

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/pilot'; CACHE=ROOT/'data/source-cache'
for p in [OUT/'images',OUT/'references',CACHE]: p.mkdir(parents=True,exist_ok=True)
rows=[]
def github_list(repository,path):
 r=requests.get(f'https://api.github.com/repos/{repository}/contents/{path}',timeout=30); r.raise_for_status()
 return sorted([x for x in r.json() if x['type']=='file'],key=lambda x:x['name'])
def raw(url):
 r=requests.get(url,timeout=60); r.raise_for_status(); return r.content

def record(task,num,data,dataset,split,url,member=None,clean=None,extra=None):
 source_sha=hashlib.sha256(data).hexdigest()
 with Image.open(io.BytesIO(data)) as im:
  if im.format!='PNG':
   stream=io.BytesIO(); im.convert('RGB').save(stream,format='PNG'); data=stream.getvalue()
 name=f'images/{task}-{num}.png'; (OUT/name).write_bytes(data)
 with Image.open(io.BytesIO(data)) as im: size=list(im.size)
 row=dict(id=f'{task}-{num}',degradation=task,source_dataset=dataset,source_split=split,source_url=url,input=name,
          input_sha256=hashlib.sha256(data).hexdigest(),source_sha256=source_sha,size=size,annotation_status='pending')
 if member: row['source_member']=member
 if clean:
  target=f'references/{task}-{num}.png'; (OUT/target).write_bytes(clean); row['clean']=target
 if extra: row.update(extra)
 rows.append(row)

# Color BSD68. Save the exact mobile PNG/noise convention, rather than silently
# treating clipped uint8 images as the paper's unquantized floating benchmark.
for i,item in enumerate(github_list('cszn/FFDNet','testsets/CBSD68')[:3],1):
 data=raw(item['download_url']); rgb=np.asarray(Image.open(io.BytesIO(data)).convert('RGB')).astype(np.float32)
 seed=41+i; noise=np.random.default_rng(seed).normal(0,25,rgb.shape)
 noisy=np.clip(np.rint(rgb+noise),0,255).astype(np.uint8); stream=io.BytesIO(); Image.fromarray(noisy).save(stream,format='PNG')
 record('denoise',i,stream.getvalue(),'CBSD68 (color BSD68)','test',item['download_url'],item['name'],data,
        dict(noise_sigma_8bit=25,noise_seed=seed,noise_convention='independent RGB Gaussian; round+clip to uint8 PNG for mobile input',source_blob_sha=item['sha']))
 print('denoise',i,flush=True)

# Exactly three low-light evaluation images included in the authors' repository.
for i,item in enumerate(github_list('weichen582/RetinexNet','data/eval/low')[:3],1):
 record('lowlight',i,raw(item['download_url']),'LOL v1','author repository eval subset',item['download_url'],item['name'],extra=dict(source_blob_sha=item['sha'],clean_reference_status='not included in repository subset'))
 print('lowlight',i,flush=True)

rain=CACHE/'Rain100L.zip'
if not rain.exists(): gdown.download(id='1-3Y7TLZ5EpYYZmFLkRKjV508kcBeibjU',output=str(rain),quiet=True)
with zipfile.ZipFile(rain) as z:
 names=sorted(n for n in z.namelist() if Path(n).name.startswith('rain-') and n.endswith('.png'))
 for i,n in enumerate(names[:3],1):
  clean='Rain100L/'+Path(n).name.replace('rain-','norain-')
  record('derain',i,z.read(n),'Rain100L','test','https://drive.google.com/file/d/1-3Y7TLZ5EpYYZmFLkRKjV508kcBeibjU/view',n,z.read(clean))
  print('derain',i,flush=True)

url='https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip?download=true'
with zipfile.ZipFile(RemoteZipReader(url)) as z:
 selected=[]; sequences=set()
 for n in sorted(z.namelist()):
  if '/test/' not in '/'+n or '/blur/' not in n or not n.endswith('.png'): continue
  sequence=n.split('/blur/')[0]
  if sequence not in sequences: selected.append(n); sequences.add(sequence)
  if len(selected)==3: break
 for i,n in enumerate(selected,1):
  record('deblur',i,z.read(n),'GoPro','test',url,n,z.read(n.replace('/blur/','/sharp/')),
         dict(dataset_page='https://seungjunnah.github.io/Datasets/gopro.html',license='CC BY 4.0'))
  print('deblur',i,flush=True)

url='https://utexas.app.box.com/index.php?rm=box_download_shared_file&shared_name=uqvnbfo68kns1210z5k5j17cvazavcd1&file_id=f_766452360255'
with zipfile.ZipFile(RemoteZipReader(url)) as z:
 selected=[]; scenes=set()
 for n in sorted(z.namelist()):
  if '/outdoor/hazy/' not in n or not n.lower().endswith(('.png','.jpg','.jpeg')): continue
  scene=Path(n).stem.split('_')[0]
  if scene not in scenes: selected.append(n); scenes.add(scene)
  if len(selected)==3: break
 for i,n in enumerate(selected,1):
  clean='SOTS/outdoor/gt/'+Path(n).stem.split('_')[0]+'.png'
  record('dehaze',i,z.read(n),'RESIDE SOTS outdoor','test',url,n,z.read(clean),
         dict(dataset_page='https://sites.google.com/view/reside-dehaze-datasets/reside-standard'))
  print('dehaze',i,flush=True)

assert Counter(row['degradation'] for row in rows)=={task:3 for task in ['denoise','derain','dehaze','deblur','lowlight']}, 'Missing task samples'
manifest=dict(version=1,scope='Unlabelled restoration pilot; current DSwinIR checkpoint is Rain100L single-task. Other degradation tasks are exploratory/OOD, not five-task reproduction.',images=rows)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print('Saved',len(rows),'samples',OUT/'manifest.json',flush=True)
