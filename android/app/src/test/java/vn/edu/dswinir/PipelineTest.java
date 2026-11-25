package vn.edu.dswinir;
import org.junit.Test;
import static org.junit.Assert.*;
import java.util.*;

public class PipelineTest {
    private Instance box(String id,int cls,float score,float x) { return new Instance(id,cls,score,new float[]{x,0,x+10,10}); }
    @Test public void tilingCoversEveryPixelExactlyOnce() {
        for (int[] dim:new int[][]{{1,1},{255,257},{256,256},{513,769}}) {
            int[] coverage=new int[dim[0]*dim[1]];
            for (int[] tile:Tiles.plan(dim[0],dim[1])) {
                assertTrue(tile[2]<=256 && tile[3]<=256);
                for (int y=0;y<tile[3];y++) for (int x=0;x<tile[2];x++) coverage[(tile[1]+y)*dim[0]+tile[0]+x]++;
            }
            for (int c:coverage) assertEquals(1,c);
        }
    }
    @Test public void perfectPredictionAndMissingPredictions() {
        Instance truth=box("a",0,1,0);
        assertArrayEquals(new double[]{1,1},MapMetric.compute(List.of(truth),List.of(box("a",0,.8f,0)),false),1e-8);
        assertArrayEquals(new double[]{0,0},MapMetric.compute(List.of(truth),List.of(),false),1e-8);
    }
    @Test public void falsePositiveBeforeTruePositiveReducesAP() {
        List<Instance> pred=List.of(box("a",0,.99f,30),box("a",0,.9f,0),box("a",0,.8f,0));
        assertArrayEquals(new double[]{.5,.5},MapMetric.compute(List.of(box("a",0,1,0)),pred,false),1e-8);
    }
    @Test public void cannotMatchAcrossImagesOrClasses() {
        List<Instance> pred=List.of(box("other",0,.9f,0),box("a",1,.8f,0));
        assertEquals(0,MapMetric.compute(List.of(box("a",0,1,0)),pred,false)[0],0);
    }
    @Test public void segmentationUsesMasksRatherThanBoxIoU() {
        Instance a=box("a",0,1,0), b=box("a",0,.9f,0);
        a.mask=new BitSet(); a.mask.set(0,10); b.mask=new BitSet(); b.mask.set(10,20);
        assertEquals(1,MapMetric.compute(List.of(a),List.of(b),false)[0],1e-8);
        assertEquals(0,MapMetric.compute(List.of(a),List.of(b),true)[0],0);
    }
    @Test(expected=IllegalArgumentException.class) public void mAPRequiresGroundTruth() { MapMetric.compute(List.of(),List.of(),false); }
    @Test public void gpuCoverageNeedsEvidence() {
        assertEquals("UNKNOWN_GPU_COVERAGE",GpuEvidence.classify("Created TensorFlow Lite delegate for GPU."));
        assertEquals("PARTIAL_GPU",GpuEvidence.classify("233 operations will run on the GPU, and the remaining 2 operations will run on the CPU."));
        assertEquals("FULL_GPU",GpuEvidence.classify("Replacing 23 out of 23 node(s) with delegate (TfLiteGpuDelegateV2) node"));
        assertEquals("UNKNOWN_GPU_COVERAGE",GpuEvidence.classify("Replacing 23 out of 23 node(s) with delegate (TfLiteXNNPackDelegate) node"));
        assertEquals("CPU_ONLY",GpuEvidence.classify("No operations will run on the GPU, and all 32 operations will run on the CPU."));
        assertEquals("FULL_GPU",GpuEvidence.classify("Replacing 32 node(s) with delegate (TfLiteGpuDelegateV2) node",32));
        assertEquals("PARTIAL_GPU",GpuEvidence.classify("Replacing 30 node(s) with delegate (TfLiteGpuDelegateV2) node",32));
        assertEquals("UNKNOWN_GPU_COVERAGE",GpuEvidence.classify("Replacing 32 node(s) with delegate (TfLiteGpuDelegateV2) node"));
    }
}
