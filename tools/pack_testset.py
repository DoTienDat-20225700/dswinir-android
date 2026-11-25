"""Validate a real annotated test set and package it for the Android app.

python tools/pack_testset.py dataset/manifest.json --out testset.zip
No images or labels are synthesized. Paths are relative to the manifest.
"""
import argparse
import json
from pathlib import Path
import zipfile

TASKS={'denoise','derain','dehaze','deblur','lowlight'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest',type=Path); p.add_argument('--out',type=Path,required=True); p.add_argument('--require-five-tasks',action='store_true'); p.add_argument('--allow-unlabelled',action='store_true'); a=p.parse_args()
    root=a.manifest.resolve().parent; data=json.loads(a.manifest.read_text())
    if data['version']!=1 or not data['images']: raise ValueError('Need version 1 and nonempty images')
    ids=set(); paths=set(); counts={name:0 for name in TASKS}
    for row in data['images']:
        if row['id'] in ids: raise ValueError('Duplicate id')
        ids.add(row['id']); counts[row['degradation']]+=1
        source=(root/row['input']).resolve()
        if not source.is_relative_to(root) or not source.is_file(): raise ValueError('Invalid image path')
        paths.add(source)
        if row.get('clean'):
            clean=(root/row['clean']).resolve()
            if not clean.is_relative_to(root) or not clean.is_file(): raise ValueError('Invalid clean reference path')
            paths.add(clean)
        if 'source_dataset' not in row or 'source_split' not in row: raise ValueError('Record source_dataset and source_split for reproducibility')
        if row.get('annotation_status')=='pending' and not a.allow_unlabelled:
            raise ValueError('Unlabelled images require --allow-unlabelled; they cannot be used to compute mAP')
        if 'annotations' not in row and row.get('annotation_status')!='pending': raise ValueError('Missing annotation status')
        for annotation in row.get('annotations',[]):
            if not 0<=annotation['class_id']<80: raise ValueError('Need COCO contiguous class 0..79')
            x1,y1,x2,y2=annotation['bbox']
            if not (0<=x1<x2 and 0<=y1<y2): raise ValueError('Invalid xyxy box')
    if a.require_five_tasks and min(counts.values())<2: raise ValueError('Need at least 2 images per task')
    if a.out.resolve() in paths or a.out.resolve()==a.manifest.resolve(): raise ValueError('Output must not overwrite an input')
    with zipfile.ZipFile(a.out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json',json.dumps(data,ensure_ascii=False,indent=2))
        for image in sorted(paths): z.write(image,image.relative_to(root).as_posix())
    print(json.dumps(counts,indent=2)); print(a.out)

if __name__=='__main__': main()
