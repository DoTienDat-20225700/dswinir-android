package vn.edu.dswinir;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import android.content.Context;
import android.graphics.Bitmap;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;
import java.io.*;
import java.util.List;

/** Fixtures are identity operators; these tests do not measure DSwinIR quality. */
@RunWith(AndroidJUnit4.class)
public class RestorationDeviceTest {
    private File fixture(String name) throws Exception {
        Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        File f=new File(context.getCacheDir(),name);
        try (InputStream in=InstrumentationRegistry.getInstrumentation().getContext().getAssets().open(name); OutputStream out=new FileOutputStream(f)) {
            byte[] b=new byte[4096]; int n; while ((n=in.read(b))!=-1) out.write(b,0,n);
        }
        return f;
    }
    @Test public void bothLayoutsPreserveRgbAndFullImageDimensions() throws Exception {
        for (String name:new String[]{"identity_nhwc.tflite","identity_nchw.tflite"}) {
            int w=357,h=259; int[] pixels=new int[w*h];
            for (int y=0;y<h;y++) for (int x=0;x<w;x++) pixels[y*w+x]=0xff000000 | ((x*7%256)<<16) | ((y*3%256)<<8) | ((x+y)%256);
            Bitmap source=Bitmap.createBitmap(pixels,w,h,Bitmap.Config.ARGB_8888);
            File model=fixture(name); assertEquals(0,ModelGraph.operatorCount(model));
            try (Restorer restorer=new Restorer(model,false)) {
                java.util.ArrayList<Integer> progress=new java.util.ArrayList<>();
                Bitmap result=restorer.restore(source,(done,total)->{ assertEquals(4,total); progress.add(done); });
                assertEquals(java.util.Arrays.asList(0,1,2,3,4),progress);
                int[] actual=new int[w*h]; result.getPixels(actual,0,w,0,0,w,h);
                assertEquals(w,result.getWidth()); assertEquals(h,result.getHeight()); assertArrayEquals(pixels,actual);
                double[] timings=restorer.benchmark(source); assertEquals(50,timings.length); for (double v:timings) assertTrue(v>=0);
                result.recycle();
            } finally { source.recycle(); }
        }
    }
    @Test(expected=IllegalArgumentException.class) public void rejectsNon256RestorationInput() throws Exception {
        try (Restorer r=new Restorer(fixture("wrong_size.tflite"),false)) { fail("Non256 model accepted"); }
    }
    @Test(expected=IOException.class) public void unlabelledImageCannotBecomeNegativeGroundTruth() throws Exception {
        org.json.JSONObject row=new org.json.JSONObject().put("annotation_status","pending");
        TestSet.annotations(row,100,100,false);
    }
    @Test public void independentYoloDecodersUndoLetterboxAndProduceInstanceMasks() throws Exception {
        Bitmap source=Bitmap.createBitmap(100,50,Bitmap.Config.ARGB_8888);
        for (boolean segment:new boolean[]{false,true}) {
            try (Yolo yolo=new Yolo(fixture(segment?"yolo_seg_fixture.tflite":"yolo_det_fixture.tflite"),segment)) {
                List<Instance> predictions=yolo.predict(source,"test"); assertEquals(1,predictions.size());
                Instance p=predictions.get(0); assertEquals(0,p.cls);
                assertArrayEquals(new float[]{25,0,75,50},p.box,1e-3f);
                if (segment) { assertNotNull(p.mask); assertEquals(2500,p.mask.cardinality()); }
                else assertNull(p.mask);
            }
        }
        source.recycle();
    }
}
