package vn.edu.dswinir;

import java.util.*;

/** 101 recall points, IoU .50:.05:.95, classes with GT only; no crowd/ignore. */
public final class MapMetric {
    public static double[] compute(List<Instance> gt,List<Instance> predictions,boolean masks) {
        Set<Integer> classes=new TreeSet<>(); for (Instance g:gt) classes.add(g.cls);
        if (classes.isEmpty()) throw new IllegalArgumentException("mAP needs human ground truth annotations");
        double ap50=0, all=0;
        for (int cls:classes) {
            List<Instance> truth=new ArrayList<>(), pred=new ArrayList<>();
            for (Instance g:gt) if (g.cls==cls) truth.add(g);
            for (Instance p:predictions) if (p.cls==cls) pred.add(p);
            pred.sort((a,b)->Float.compare(b.score,a.score));
            for (int t=0;t<10;t++) {
                double threshold=.5+.05*t;
                boolean[] used=new boolean[truth.size()];
                double[] precision=new double[pred.size()], recall=new double[pred.size()];
                int tp=0;
                for (int i=0;i<pred.size();i++) {
                    Instance p=pred.get(i); int match=-1; double best=threshold-1e-9;
                    for (int j=0;j<truth.size();j++) if (!used[j] && p.image.equals(truth.get(j).image)) {
                        double overlap=Instance.iou(p,truth.get(j),masks);
                        if (overlap>best) { best=overlap; match=j; }
                    }
                    if (match>=0) { used[match]=true; tp++; }
                    precision[i]=(double)tp/(i+1); recall[i]=(double)tp/truth.size();
                }
                for (int i=precision.length-2;i>=0;i--) precision[i]=Math.max(precision[i],precision[i+1]);
                double ap=0;
                for (int r=0;r<=100;r++) {
                    double value=0;
                    for (int i=0;i<recall.length;i++) if (recall[i]+1e-9>=r/100.0) { value=precision[i]; break; }
                    ap+=value/101;
                }
                if (t==0) ap50+=ap;
                all+=ap;
            }
        }
        return new double[]{ap50/classes.size(),all/(classes.size()*10)};
    }
    private MapMetric() {}
}
