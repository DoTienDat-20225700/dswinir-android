package vn.edu.dswinir;
import java.util.BitSet;

public final class Instance {
    public final String image;
    public final int cls;
    public final float score;
    public final float[] box;
    public BitSet mask;
    public float[] coefficients;
    public Instance(String image,int cls,float score,float[] box) {
        this.image=image; this.cls=cls; this.score=score; this.box=box;
    }
    public static double iou(Instance a, Instance b, boolean masks) {
        if (masks) {
            if (a.mask==null || b.mask==null) throw new IllegalArgumentException("Missing instance mask");
            BitSet v=(BitSet)a.mask.clone(); v.and(b.mask);
            int intersection=v.cardinality();
            int union=a.mask.cardinality()+b.mask.cardinality()-intersection;
            return union==0?0:(double)intersection/union;
        }
        float[] x=a.box, y=b.box;
        double inter=Math.max(0,Math.min(x[2],y[2])-Math.max(x[0],y[0])) * Math.max(0,Math.min(x[3],y[3])-Math.max(x[1],y[1]));
        double union=Math.max(0,x[2]-x[0])*Math.max(0,x[3]-x[1])+Math.max(0,y[2]-y[0])*Math.max(0,y[3]-y[1])-inter;
        return union>0?inter/union:0;
    }
}
