package vn.edu.dswinir;

import android.graphics.*;
import org.tensorflow.lite.*;
import java.io.File;
import java.nio.*;
import java.util.*;

/** Ultralytics YOLO11 FP32 TFLite, nms=False, batch=1, normalized xywh output.
 * Detection and instance segmentation use independent models and inferences.
 */
public final class Yolo implements AutoCloseable {
    private final Interpreter model;
    private final boolean segment;
    private final int size, candidates, channels, masks, detIndex, protoIndex;
    private final int[] protoShape;
    private final ByteBuffer input, detections, prototypes;
    public Yolo(File file,boolean segment) {
        this.segment=segment;
        model=new Interpreter(file,new Interpreter.Options().setNumThreads(Restorer.cpuThreads()));
        try {
            model.allocateTensors();
            int[] s=model.getInputTensor(0).shape();
            if (model.getInputTensorCount()!=1 || s.length!=4 || s[0]!=1 || s[1]!=s[2] || s[3]!=3)
                throw new IllegalArgumentException("YOLO input must be [1,S,S,3]");
            if (model.getOutputTensorCount()!=(segment?2:1)) throw new IllegalArgumentException("Wrong YOLO task/model");
            if (model.getInputTensor(0).dataType()!=DataType.FLOAT32) throw new IllegalArgumentException("YOLO must be FP32");
            size=s[1]; int di=-1, pi=-1;
            for (int i=0;i<model.getOutputTensorCount();i++) {
                if (model.getOutputTensor(i).dataType()!=DataType.FLOAT32) throw new IllegalArgumentException("YOLO output must be FP32");
                if (model.getOutputTensor(i).shape().length==3) di=i;
                if (model.getOutputTensor(i).shape().length==4) pi=i;
            }
            if (di<0 || (segment && pi<0)) throw new IllegalArgumentException("Expected raw YOLO outputs, export nms=False");
            detIndex=di; protoIndex=pi;
            int[] ds=model.getOutputTensor(di).shape();
            channels=ds[1]; candidates=ds[2]; masks=segment?32:0;
            if (ds[0]!=1 || channels!=84+masks) throw new IllegalArgumentException("Use COCO 80-class YOLO11 raw [1,84/116,N]");
            protoShape=segment?model.getOutputTensor(pi).shape():null;
            if (segment && (protoShape[0]!=1 || protoShape[3]!=32)) throw new IllegalArgumentException("Prototype must be NHWC [1,H,W,32]");
            input=buf(size*size*3*4); detections=buf(model.getOutputTensor(di).numBytes());
            prototypes=segment?buf(model.getOutputTensor(pi).numBytes()):null;
        } catch (RuntimeException | Error e) { model.close(); throw e; }
    }
    private static ByteBuffer buf(int bytes) { return ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()); }
    private float at(int c,int n) { return detections.getFloat((c*candidates+n)*4); }
    public List<Instance> predict(Bitmap image,String id) {
        float scale=Math.min((float)size/image.getWidth(),(float)size/image.getHeight());
        int nw=Math.round(image.getWidth()*scale), nh=Math.round(image.getHeight()*scale);
        int left=(size-nw)/2, top=(size-nh)/2;
        Bitmap boxed=Bitmap.createBitmap(size,size,Bitmap.Config.ARGB_8888);
        Canvas canvas=new Canvas(boxed); canvas.drawColor(Color.rgb(114,114,114));
        canvas.drawBitmap(image,null,new Rect(left,top,left+nw,top+nh),new Paint(Paint.FILTER_BITMAP_FLAG));
        int[] pixels=new int[size*size]; boxed.getPixels(pixels,0,size,0,0,size,size); boxed.recycle();
        input.clear(); for (int p:pixels) for (int c=0;c<3;c++) input.putFloat(((p>>(16-8*c))&255)/255f);
        input.rewind(); detections.clear();
        Map<Integer,Object> outs=new HashMap<>(); outs.put(detIndex,detections);
        if (segment) { prototypes.clear(); outs.put(protoIndex,prototypes); }
        model.runForMultipleInputsOutputs(new Object[]{input},outs);
        List<Instance> raw=new ArrayList<>();
        for (int n=0;n<candidates;n++) {
            int cls=0; float score=at(4,n);
            for (int c=1;c<80;c++) if (at(4+c,n)>score) { score=at(4+c,n); cls=c; }
            if (!Float.isFinite(score) || score<.001f) continue;
            float cx=at(0,n)*size, cy=at(1,n)*size, w=at(2,n)*size, h=at(3,n)*size;
            float[] b={Math.max(0,(cx-w/2-left)/scale),Math.max(0,(cy-h/2-top)/scale),Math.min(image.getWidth(),(cx+w/2-left)/scale),Math.min(image.getHeight(),(cy+h/2-top)/scale)};
            if (b[2]<=b[0] || b[3]<=b[1]) continue;
            Instance p=new Instance(id,cls,score,b);
            if (segment) { p.coefficients=new float[32]; for (int c=0;c<32;c++) p.coefficients[c]=at(84+c,n); }
            raw.add(p);
        }
        raw.sort((a,b)->Float.compare(b.score,a.score));
        List<Instance> kept=new ArrayList<>();
        for (Instance p:raw) {
            boolean overlap=false;
            for (Instance k:kept) if (p.cls==k.cls && Instance.iou(p,k,false)>.7) { overlap=true; break; }
            if (!overlap) { kept.add(p); if (kept.size()==300) break; }
        }
        if (segment) for (Instance p:kept) mask(p,image.getWidth(),image.getHeight(),scale,left,top);
        return kept;
    }
    private void mask(Instance p,int width,int height,float scale,int left,int top) {
        int ph=protoShape[1], pw=protoShape[2];
        float[] logits=new float[ph*pw];
        for (int i=0;i<logits.length;i++) for (int c=0;c<32;c++) logits[i]+=prototypes.getFloat((i*32+c)*4)*p.coefficients[c];
        p.mask=new BitSet(width*height);
        for (int y=Math.max(0,(int)Math.ceil(p.box[1]));y<Math.min(height,(int)Math.ceil(p.box[3]));y++)
            for (int x=Math.max(0,(int)Math.ceil(p.box[0]));x<Math.min(width,(int)Math.ceil(p.box[2]));x++) {
                float u=((x+.5f)*scale+left)*pw/size-.5f, v=((y+.5f)*scale+top)*ph/size-.5f;
                u=Math.max(0,Math.min(pw-1,u)); v=Math.max(0,Math.min(ph-1,v));
                int x0=(int)u,y0=(int)v,x1=Math.min(pw-1,x0+1),y1=Math.min(ph-1,y0+1);
                float dx=u-x0,dy=v-y0;
                float logit=(1-dy)*((1-dx)*logits[y0*pw+x0]+dx*logits[y0*pw+x1])+dy*((1-dx)*logits[y1*pw+x0]+dx*logits[y1*pw+x1]);
                if (logit>0) p.mask.set(y*width+x);
            }
        p.coefficients=null;
    }
    public static Bitmap draw(Bitmap image,List<Instance> predictions) {
        Bitmap b=image.copy(Bitmap.Config.ARGB_8888,true); Canvas c=new Canvas(b);
        Paint paint=new Paint(); paint.setStrokeWidth(3); paint.setTextSize(24);
        for (Instance p:predictions) if (p.score>=.25f) {
            if (p.mask!=null) {
                int[] overlay=new int[b.getWidth()*b.getHeight()];
                for (int i=p.mask.nextSetBit(0);i>=0;i=p.mask.nextSetBit(i+1)) overlay[i]=0x6600bb88;
                Bitmap m=Bitmap.createBitmap(overlay,b.getWidth(),b.getHeight(),Bitmap.Config.ARGB_8888); c.drawBitmap(m,0,0,null); m.recycle();
            }
            paint.setColor(Color.rgb(0,160,100)); paint.setStyle(Paint.Style.STROKE);
            c.drawRect(p.box[0],p.box[1],p.box[2],p.box[3],paint); paint.setStyle(Paint.Style.FILL);
            c.drawText(p.cls+String.format(java.util.Locale.US," %.2f",p.score),p.box[0],Math.max(24,p.box[1]),paint);
        }
        return b;
    }
    @Override public void close() { model.close(); }
}
