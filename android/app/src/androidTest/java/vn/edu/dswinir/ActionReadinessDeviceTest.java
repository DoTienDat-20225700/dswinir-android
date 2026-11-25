package vn.edu.dswinir;

import android.app.Activity;
import android.content.*;
import android.graphics.Bitmap;
import android.net.Uri;
import android.view.*;
import android.widget.Button;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.*;
import org.junit.runner.RunWith;
import org.json.*;
import java.io.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.*;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ActionReadinessDeviceTest {
    private Button button(View view,String label) {
        if (view instanceof Button && ((Button)view).getText().toString().equals(label)) return (Button)view;
        if (view instanceof ViewGroup) for (int i=0;i<((ViewGroup)view).getChildCount();i++) {
            Button b=button(((ViewGroup)view).getChildAt(i),label); if (b!=null) return b;
        }
        return null;
    }
    @Test public void choosingSingleImageEnablesRestoreWithoutEnablingDatasetMap() throws Exception {
        android.app.Instrumentation inst=InstrumentationRegistry.getInstrumentation(); Context context=inst.getTargetContext();
        File model=new File(context.getFilesDir(),"restoration.tflite"); boolean seeded=!model.exists();
        if (seeded) try (InputStream in=inst.getContext().getAssets().open("identity_nhwc.tflite"); OutputStream out=new FileOutputStream(model)) {
            byte[] bytes=new byte[4096]; int n; while ((n=in.read(bytes))!=-1) out.write(bytes,0,n);
        }
        File photo=new File(context.getCacheDir(),"ui-test-input.png"); Bitmap image=Bitmap.createBitmap(17,19,Bitmap.Config.ARGB_8888);
        try (OutputStream out=new FileOutputStream(photo)) { image.compress(Bitmap.CompressFormat.PNG,100,out); } image.recycle();
        File session=new File(context.getFilesDir(),"last-session");
        File backup=new File(context.getCacheDir(),"session-backup-"+System.nanoTime());
        boolean hadSession=session.exists(); if (hadSession) java.nio.file.Files.move(session.toPath(),backup.toPath());
        MainActivity activity=(MainActivity)inst.startActivitySync(new Intent(context,MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
        try {
            inst.runOnMainSync(()->{
                assertFalse(button(activity.getWindow().getDecorView(),"Bắt đầu khôi phục").isEnabled());
                assertFalse(button(activity.getWindow().getDecorView(),"Chấm điểm tìm vật thể").isEnabled());
                activity.onActivityResult(4,Activity.RESULT_OK,new Intent().setData(Uri.fromFile(photo)));
            });
            AtomicBoolean enabled=new AtomicBoolean();
            for (int i=0;i<100 && !enabled.get();i++) {
                inst.runOnMainSync(()->{ Button b=button(activity.getWindow().getDecorView(),"Bắt đầu khôi phục"); enabled.set(b!=null && b.isEnabled()); });
                if (!enabled.get()) Thread.sleep(50);
            }
            assertTrue("Photo should enable restoration without any dataset ZIP",enabled.get());
            inst.runOnMainSync(()->{
                assertTrue("Restore action must stay visible after choosing a photo",button(activity.getWindow().getDecorView(),"Bắt đầu khôi phục").getGlobalVisibleRect(new android.graphics.Rect()));
                assertFalse(button(activity.getWindow().getDecorView(),"Chấm điểm tìm vật thể").isEnabled());
                assertFalse(button(activity.getWindow().getDecorView(),"Chấm điểm tô vùng vật thể").isEnabled());
                assertFalse(button(activity.getWindow().getDecorView(),"Tìm và khoanh vật thể").isEnabled());
            });
        } finally { inst.runOnMainSync(activity::finish); if (seeded) model.delete(); photo.delete();
            if (session.isDirectory()) { for (File f:session.listFiles()) f.delete(); session.delete(); }
            if (hadSession) java.nio.file.Files.move(backup.toPath(),session.toPath());
        }
    }
    @Test public void unlabelledBboxAndMaskDatasetsHaveDistinctPrerequisites() throws Exception {
        Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        for (int state=0;state<3;state++) {
            JSONObject row=new JSONObject().put("id","sample").put("degradation","derain").put("input","sample.png");
            if (state==0) row.put("annotation_status","pending");
            else {
                JSONObject annotation=new JSONObject().put("class_id",0).put("bbox",new JSONArray("[0,0,10,10]"));
                if (state==2) annotation.put("polygon",new JSONArray("[0,0,10,0,10,10]"));
                row.put("annotations",new JSONArray().put(annotation));
            }
            ByteArrayOutputStream data=new ByteArrayOutputStream();
            try (ZipOutputStream z=new ZipOutputStream(data)) {
                z.putNextEntry(new ZipEntry("manifest.json")); z.write(new JSONObject().put("version",1).put("images",new JSONArray().put(row)).toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)); z.closeEntry();
                z.putNextEntry(new ZipEntry("sample.png")); z.write(new byte[]{0}); z.closeEntry();
            }
            TestSet set=TestSet.unpack(new ByteArrayInputStream(data.toByteArray()),new File(context.getCacheDir(),"readiness-"+System.nanoTime()));
            assertEquals(state>0,set.hasGroundTruth(false)); assertEquals(state==2,set.hasGroundTruth(true));
        }
    }
}
