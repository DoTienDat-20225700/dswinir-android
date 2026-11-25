import importlib
for p in ['onnx', 'onnxruntime', 'tensorflow', 'keras', 'onnx2tf', 'onnxconverter_common']:
    try:
        m = importlib.import_module(p)
        print(p, getattr(m, '__version__', '?'), 'OK')
    except Exception as e:
        print(p, 'MISSING:', type(e).__name__, str(e)[:80])
