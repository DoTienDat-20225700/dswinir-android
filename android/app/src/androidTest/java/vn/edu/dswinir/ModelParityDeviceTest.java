package vn.edu.dswinir;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import android.content.Context;
import org.json.*;
import org.junit.*;
import org.junit.runner.RunWith;
import org.tensorflow.lite.Interpreter;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import java.util.*;
import static org.junit.Assert.*;

/** Optional numerical verification of REAL converted models, not mAP. */
@RunWith(AndroidJUnit4.class)
public class ModelParityDeviceTest {
    private ByteBuffer buffer(File path) throws Exception {
        byte[] bytes=Files.readAllBytes(path.toPath());
        ByteBuffer b=ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder()); b.put(bytes); b.rewind(); return b;
    }
    @Test public void realModelsMatchTorchFixturesOnAndroid() throws Exception {
        String path=InstrumentationRegistry.getArguments().getString("models_dir");
        Assume.assumeTrue("Pass -e models_dir with staged real model fixtures",path!=null);
        File root=new File(path);
        JSONObject manifest=new JSONObject(new String(Files.readAllBytes(new File(root,"manifest.json").toPath()),java.nio.charset.StandardCharsets.UTF_8));
        JSONArray cases=manifest.getJSONArray("cases"), results=new JSONArray();
        for (int i=0;i<cases.length();i++) {
            JSONObject row=cases.getJSONObject(i); String name=row.getString("name");
            Interpreter.Options options=new Interpreter.Options().setNumThreads(4).setUseXNNPACK(!name.startsWith("restoration"));
            try (Interpreter interpreter=new Interpreter(new File(root,row.getString("model")),options)) {
                interpreter.allocateTensors(); ByteBuffer input=buffer(new File(root,row.getString("input")));
                Map<Integer,Object> outputs=new HashMap<>();
                for (int j=0;j<interpreter.getOutputTensorCount();j++) outputs.put(j,ByteBuffer.allocateDirect(interpreter.getOutputTensor(j).numBytes()).order(ByteOrder.nativeOrder()));
                long start=android.os.SystemClock.elapsedRealtimeNanos();
                interpreter.runForMultipleInputsOutputs(new Object[]{input},outputs);
                double ms=(android.os.SystemClock.elapsedRealtimeNanos()-start)/1e6;
                JSONArray reference=row.getJSONArray("outputs"), errors=new JSONArray();
                for (int j=0;j<reference.length();j++) {
                    JSONObject ref=reference.getJSONObject(j); JSONArray dims=ref.getJSONArray("shape"); int[] shape=new int[dims.length()];
                    for (int k=0;k<shape.length;k++) shape[k]=dims.getInt(k);
                    int index=-1; for (int k=0;k<interpreter.getOutputTensorCount();k++) if (Arrays.equals(shape,interpreter.getOutputTensor(k).shape())) index=k;
                    assertTrue(name+": output shape missing",index>=0);
                    ByteBuffer actual=(ByteBuffer)outputs.get(index), expected=buffer(new File(root,ref.getString("file")));
                    assertEquals(expected.capacity(),actual.capacity()); double maximum=0;
                    for (int k=0;k<actual.capacity()/4;k++) {
                        float got=actual.getFloat(k*4), want=expected.getFloat(k*4); assertTrue(name+": nonfinite",Float.isFinite(got));
                        maximum=Math.max(maximum,Math.abs((double)got-want));
                    }
                    errors.put(maximum); assertTrue(name+": max abs "+maximum,maximum<=1e-3);
                }
                results.put(new JSONObject().put("name",name).put("max_abs",errors).put("single_invoke_ms",ms));
                System.out.println("MODEL_PARITY "+name+" "+errors+" ms="+ms);
            }
        }
        Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        File report=new File(context.getExternalFilesDir(null),"model-parity.json");
        JSONObject out=new JSONObject().put("android_api",android.os.Build.VERSION.SDK_INT).put("device",android.os.Build.MODEL)
            .put("scope","numerical parity on synthetic Torch fixtures; not quality/mAP or 5+50 benchmark").put("cases",results);
        Files.write(report.toPath(),out.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }
}
