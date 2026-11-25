package vn.edu.dswinir;

import android.graphics.*;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;

/** A ZIP with manifest.json, test images and manual COCO-class annotations. */
public final class TestSet {
    public static final Set<String> TASKS = Collections.unmodifiableSet(new HashSet<>(Arrays.asList("denoise","derain","dehaze","deblur","lowlight")));
    public final File root;
    public final JSONArray images;
    private TestSet(File root,JSONArray images) { this.root=root; this.images=images; }
    public boolean hasGroundTruth(boolean masks) {
        try {
            for (int i=0;i<images.length();i++) {
                JSONObject row=images.getJSONObject(i);
                if (row.optString("annotation_status").equals("pending") || !row.has("annotations")) return false;
                JSONArray annotations=row.getJSONArray("annotations");
                if (masks) for (int j=0;j<annotations.length();j++) {
                    JSONArray polygon=annotations.getJSONObject(j).optJSONArray("polygon");
                    if (polygon==null || polygon.length()<6 || polygon.length()%2!=0) return false;
                }
            }
            return images.length()>0;
        } catch (JSONException e) { return false; }
    }
    public File file(String name) throws IOException {
        File f=new File(root,name).getCanonicalFile();
        if (!f.getPath().startsWith(root.getCanonicalPath()+File.separator) || !f.isFile()) throw new IOException("Invalid dataset path: "+name);
        return f;
    }
    public static TestSet unpack(InputStream input,File root) throws Exception {
        if (!root.mkdirs()) throw new IOException("Cannot create dataset folder");
        long total=0; int entries=0;
        try (ZipInputStream zip=new ZipInputStream(input)) {
            ZipEntry e; byte[] b=new byte[65536];
            while ((e=zip.getNextEntry())!=null) {
                if (++entries>20000) throw new IOException("Too many ZIP entries");
                File target=new File(root,e.getName()).getCanonicalFile();
                if (!target.getPath().startsWith(root.getCanonicalPath()+File.separator)) throw new IOException("Invalid ZIP path");
                if (e.isDirectory()) { target.mkdirs(); continue; }
                target.getParentFile().mkdirs();
                try (OutputStream out=new FileOutputStream(target)) {
                    int n; while ((n=zip.read(b))!=-1) { total+=n; if (total>1024L*1024*1024) throw new IOException("Dataset exceeds 1 GiB"); out.write(b,0,n); }
                }
            }
        }
        JSONObject manifest=new JSONObject(new String(java.nio.file.Files.readAllBytes(new File(root,"manifest.json").toPath()),StandardCharsets.UTF_8));
        if (manifest.getInt("version")!=1) throw new IOException("Unsupported manifest version");
        JSONArray images=manifest.getJSONArray("images"); if (images.length()==0) throw new IOException("Empty test set");
        Set<String> ids=new HashSet<>();
        TestSet set=new TestSet(root,images);
        for (int i=0;i<images.length();i++) {
            JSONObject row=images.getJSONObject(i);
            if (!ids.add(row.getString("id"))) throw new IOException("Duplicate image id");
            if (!TASKS.contains(row.getString("degradation"))) throw new IOException("Invalid degradation");
            set.file(row.getString("input"));
            if (!row.optString("annotation_status").equals("pending")) row.getJSONArray("annotations");
        }
        return set;
    }
    public static List<Instance> annotations(JSONObject row,int width,int height,boolean masks) throws Exception {
        if (row.optString("annotation_status").equals("pending")) throw new IOException("Ảnh chưa có nhãn chuẩn; chưa thể tính mAP");
        List<Instance> gt=new ArrayList<>(); JSONArray items=row.getJSONArray("annotations");
        for (int i=0;i<items.length();i++) {
            JSONObject a=items.getJSONObject(i); int cls=a.getInt("class_id");
            if (cls<0 || cls>=80) throw new IOException("Use contiguous COCO classes 0..79, not category ids 1..90");
            JSONArray box=a.getJSONArray("bbox"); if (box.length()!=4) throw new IOException("bbox must be xyxy");
            float[] b=new float[4]; for (int j=0;j<4;j++) b[j]=(float)box.getDouble(j);
            if (!(b[0]>=0 && b[1]>=0 && b[2]>b[0] && b[3]>b[1] && b[2]<=width && b[3]<=height)) throw new IOException("Invalid bbox");
            Instance item=new Instance(row.getString("id"),cls,1,b);
            if (masks) {
                JSONArray polygon=a.getJSONArray("polygon");
                if (polygon.length()<6 || polygon.length()%2!=0) throw new IOException("polygon needs >=3 xy pairs");
                Bitmap bm=Bitmap.createBitmap(width,height,Bitmap.Config.ARGB_8888);
                Path path=new Path();
                for (int j=0;j<polygon.length();j+=2) {
                    float x=(float)polygon.getDouble(j), y=(float)polygon.getDouble(j+1);
                    if (!Float.isFinite(x) || !Float.isFinite(y) || x<0 || y<0 || x>width || y>height) throw new IOException("Invalid polygon");
                    if (j==0) path.moveTo(x,y); else path.lineTo(x,y);
                }
                path.close(); Paint paint=new Paint(); paint.setColor(Color.WHITE); new Canvas(bm).drawPath(path,paint);
                int[] pixels=new int[width*height]; bm.getPixels(pixels,0,width,0,0,width,height); bm.recycle();
                item.mask=new BitSet(pixels.length); for (int j=0;j<pixels.length;j++) if (pixels[j]!=0) item.mask.set(j);
            }
            gt.add(item);
        }
        return gt;
    }
}
