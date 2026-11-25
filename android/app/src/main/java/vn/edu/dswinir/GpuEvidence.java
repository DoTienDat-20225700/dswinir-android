package vn.edu.dswinir;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Never infer GPU coverage from successful delegate creation or latency. */
public final class GpuEvidence {
    public static String classify(String logs) {
        return classify(logs,-1);
    }
    public static String classify(String logs,int singleGraphOperatorCount) {
        Matcher partial = Pattern.compile("(\\d+) operations will run on the GPU, and the remaining (\\d+) operations will run on the CPU").matcher(logs);
        if (partial.find()) {
            int gpu=Integer.parseInt(partial.group(1)), cpu=Integer.parseInt(partial.group(2));
            return gpu==0 ? "CPU_ONLY" : cpu==0 ? "FULL_GPU" : "PARTIAL_GPU";
        }
        if (logs.contains("No operations will run on the GPU")) return "CPU_ONLY";
        Matcher full=Pattern.compile("Replacing (\\d+) out of (\\d+) node\\(s\\) with delegate \\(TfLiteGpuDelegate[^)]*\\)").matcher(logs);
        if (full.find() && full.group(1).equals(full.group(2)) && !full.group(1).equals("0")) return "FULL_GPU";
        Matcher old=Pattern.compile("Replacing (\\d+) node\\(s\\) with delegate \\(TfLiteGpuDelegate[^)]*\\)").matcher(logs);
        if (old.find() && singleGraphOperatorCount>0) {
            int replaced=Integer.parseInt(old.group(1));
            if (replaced==singleGraphOperatorCount) return "FULL_GPU";
            if (replaced>0 && replaced<singleGraphOperatorCount) return "PARTIAL_GPU";
        }
        return "UNKNOWN_GPU_COVERAGE";
    }
    private GpuEvidence() {}
}
