"""Package the local conversion source for Colab; excludes weights/secrets."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
FILES=['model.py','natten_pure.py','validate_model.py','tools/restore_tiled.py',
       'convert/prepare_tf_weights.py','convert/direct_tf.py']
with ZipFile(ROOT/'dswinir-conversion.zip','w',ZIP_DEFLATED) as archive:
    for path in FILES: archive.write(ROOT/path,path)
print(ROOT/'dswinir-conversion.zip')
