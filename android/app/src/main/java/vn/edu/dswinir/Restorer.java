package vn.edu.dswinir;

import android.graphics.Bitmap;
import android.os.SystemClock;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CancellationException;

/** Create, invoke and close on one worker thread, including the GPU delegate. */
public final class Restorer implements AutoCloseable {
    private Interpreter interpreter;
    private volatile boolean cancelled;
    public static int cpuThreads() { return Math.max(1,Math.min(4,Runtime.getRuntime().availableProcessors()-1)); }
    private GpuDelegate delegate;
    private ByteBuffer input, output;
    private boolean nchwIn, nchwOut;
    public final boolean gpuRequested;
    public Restorer(File model, boolean gpu) {
        gpuRequested = gpu;
        try {
            if (ModelGraph.operatorCount(model)<0) throw new IllegalArgumentException("Restoration model must have one static subgraph");
            // This attention graph creates hundreds of XNNPACK partitions.
            // Their separate arenas exhausted a 2 GiB Android10 emulator.
            // Builtin CPU kernels share the interpreter's tensor arena.
            Interpreter.Options options = new Interpreter.Options().setNumThreads(cpuThreads()).setUseXNNPACK(false).setCancellable(true);
            if (gpu) {
                delegate = new GpuDelegate(new GpuDelegate.Options().setPrecisionLossAllowed(false));
                options.addDelegate(delegate);
            }
            interpreter = new Interpreter(model, options);
            interpreter.allocateTensors();
            if (interpreter.getInputTensorCount()!=1 || interpreter.getOutputTensorCount()!=1)
                throw new IllegalArgumentException("DSwinIR requires exactly one input/output");
            nchwIn = layout(interpreter.getInputTensor(0).shape());
            nchwOut = layout(interpreter.getOutputTensor(0).shape());
            if (interpreter.getInputTensor(0).dataType()!=DataType.FLOAT32 || interpreter.getOutputTensor(0).dataType()!=DataType.FLOAT32)
                throw new IllegalArgumentException("Use an FP32 model");
            input = buffer(); output = buffer();
        } catch (RuntimeException | Error e) { close(); throw e; }
    }
    private static ByteBuffer buffer() {
        return ByteBuffer.allocateDirect(256*256*3*4).order(ByteOrder.nativeOrder());
    }
    private static boolean layout(int[] s) {
        if (Arrays.equals(s, new int[]{1,3,256,256})) return true;
        if (Arrays.equals(s, new int[]{1,256,256,3})) return false;
        throw new IllegalArgumentException("Expected [1,256,256,3] or [1,3,256,256], got " + Arrays.toString(s));
    }
    private void encode(Bitmap image, int left, int top, int width, int height) {
        int[] pixels = new int[width*height];
        image.getPixels(pixels,0,width,left,top,width,height);
        input.clear();
        for (int c=0;c<(nchwIn?3:1);c++) for (int y=0;y<256;y++) for (int x=0;x<256;x++) {
            int p = pixels[Math.min(y,height-1)*width+Math.min(x,width-1)];
            if (nchwIn) input.putFloat(((p>>(16-c*8))&255)/255f);
            else for (int k=0;k<3;k++) input.putFloat(((p>>(16-k*8))&255)/255f);
        }
        input.rewind();
    }
    public synchronized void cancel() {
        cancelled=true;
        if (interpreter!=null) interpreter.setCancelled(true);
    }
    private void checkCancelled() { if(cancelled) throw new CancellationException("Restoration cancelled"); }
    private void invoke() {
        checkCancelled();
        input.rewind(); output.clear();
        try { interpreter.run(input,output); } catch (RuntimeException e) { checkCancelled(); throw e; }
        checkCancelled(); output.rewind();
    }
    public double[] benchmark(Bitmap image) { return benchmark(image,(done,total)->{}); }
    public double[] benchmark(Bitmap image,Progress progress) {
        progress.update(0,55);
        encode(image,0,0,Math.min(image.getWidth(),256),Math.min(image.getHeight(),256));
        for (int i=0;i<5;i++) { invoke(); progress.update(i+1,55); }
        double[] ms = new double[50];
        for (int i=0;i<50;i++) {
            long start = SystemClock.elapsedRealtimeNanos(); invoke();
            ms[i]=(SystemClock.elapsedRealtimeNanos()-start)/1e6;
            progress.update(i+6,55);
        }
        return ms;
    }
    public Bitmap restore(Bitmap image) {
        return restore(image,(done,total)->{});
    }
    public interface Progress { void update(int completed,int total); }
    public Bitmap restore(Bitmap image,Progress progress) {
        Bitmap result = Bitmap.createBitmap(image.getWidth(), image.getHeight(),Bitmap.Config.ARGB_8888);
        List<int[]> tiles=Tiles.plan(image.getWidth(),image.getHeight()); int completed=0;
        progress.update(0,tiles.size());
        try {
        for (int[] tile : tiles) {
            checkCancelled();
            encode(image,tile[0],tile[1],tile[2],tile[3]); invoke();
            int[] pixels = new int[tile[2]*tile[3]];
            for (int y=0;y<tile[3];y++) for (int x=0;x<tile[2];x++) {
                int p=0xff000000;
                for (int c=0;c<3;c++) {
                    int pos = nchwOut ? c*256*256+y*256+x : (y*256+x)*3+c;
                    float v=output.getFloat(pos*4);
                    if (!Float.isFinite(v)) { result.recycle(); throw new IllegalStateException("Non-finite model output"); }
                    p |= Math.max(0,Math.min(255,Math.round(v*255))) << (16-c*8);
                }
                pixels[y*tile[2]+x]=p;
            }
            result.setPixels(pixels,0,tile[2],tile[0],tile[1],tile[2],tile[3]);
            progress.update(++completed,tiles.size());
        }
        return result;
        } catch (RuntimeException | Error e) { if(!result.isRecycled()) result.recycle(); throw e; }
    }
    @Override public synchronized void close() {
        if (interpreter!=null) { interpreter.close(); interpreter=null; }
        if (delegate!=null) { delegate.close(); delegate=null; }
    }
}
