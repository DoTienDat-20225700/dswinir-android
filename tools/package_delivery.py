"""Package the concrete APK, validated models, pilot inputs and evidence."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'DSwinIRLab.apk':'android/app/build/outputs/apk/debug/app-debug.apk',
 'models/dswinir_256_fp32.tflite':'convert/prepared256/dswinir_256_fp32.tflite',
 'models/yolo11n_fp32.tflite':'convert/yolo/detection/yolo11n_fp32.tflite',
 'models/yolo11n-seg_fp32.tflite':'convert/yolo/segmentation/yolo11n-seg_fp32.tflite',
 'testset-15-unlabelled.zip':'deliverables/testset-15-unlabelled.zip',
 'README.txt':'deliverables/README.txt',
}
for source in (ROOT/'docs/verification').glob('*'):
 if source.suffix in ['.json','.png','.txt','.zip']: FILES['reports/'+source.name]=str(source.relative_to(ROOT))
for source in (ROOT/'docs/verification/rain100l-example').glob('*'):
 if source.is_file(): FILES['example/'+source.name]=str(source.relative_to(ROOT))
for name in ['README.md','model.py','natten_pure.py','validate_model.py','download_weights.sh','docs/implementation-status.md','docs/testset-format.json']:
 FILES['source/'+name]=name
for folder in ['tools','tests','android/app/src','android/gradle/wrapper']:
 for f in (ROOT/folder).rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts:
   name=str(f.relative_to(ROOT)); FILES['source/'+name]=name
for name in ['.vscode/tasks.json','android/settings.gradle','android/build.gradle','android/gradle.properties','android/gradlew','android/gradlew.bat','android/app/build.gradle',
             'convert/direct_tf.py','convert/prepare_tf_weights.py','convert/prepare_yolo.py','convert/convert_yolo.py','convert/DSwinIR_Colab.ipynb']:
 FILES['source/'+name]=name
path=ROOT/'deliverables/DSwinIR-Android-Lab.zip'
with ZipFile(path,'w',compression=ZIP_DEFLATED,compresslevel=4) as z:
 for dest,source in FILES.items():
  f=ROOT/source
  if not f.is_file(): raise FileNotFoundError(f)
  z.write(f,dest,compress_type=ZIP_STORED if f.suffix in ['.zip','.png'] else ZIP_DEFLATED)
print(path,'bytes',path.stat().st_size)
