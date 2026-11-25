package vn.edu.dswinir;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import org.json.JSONObject;
import java.io.File;
import java.security.MessageDigest;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Locale;
import java.util.TreeSet;

/**
 * Reusable Swiss Optical Lab UI components: section headers, status pills,
 * mAP delta tables, benchmark comparison bars, and model inventory rows.
 */
public final class StudioWidgets {
    private StudioWidgets() {}

    public static View hairline(Context ctx, StudioTheme t, int marginTopDp, int marginBottomDp) {
        View v = new View(ctx);
        v.setBackgroundColor(t.hairline);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Math.max(1, t.dp(1)));
        lp.setMargins(0, t.dp(marginTopDp), 0, t.dp(marginBottomDp));
        v.setLayoutParams(lp);
        return v;
    }

    public static LinearLayout sectionBlock(Context ctx, StudioTheme t, LinearLayout parent,
                                            String indexTag, String title, String subtitle) {
        LinearLayout box = new LinearLayout(ctx);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(t.dp(16), t.dp(14), t.dp(16), t.dp(16));
        box.setBackground(t.outlined(t.surface, t.hairline, 1, 14));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, 0, 0, t.dp(12));
        parent.addView(box, lp);

        LinearLayout header = new LinearLayout(ctx);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        box.addView(header);

        if (indexTag != null && !indexTag.isEmpty()) {
            TextView tag = t.mono(ctx, indexTag, 10.5f, t.tealSoftInk);
            tag.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
            tag.setBackground(t.rect(t.tealSoft, 6));
            tag.setPadding(t.dp(7), t.dp(2), t.dp(7), t.dp(2));
            LinearLayout.LayoutParams tagLp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            tagLp.setMarginEnd(t.dp(8));
            header.addView(tag, tagLp);
        }

        TextView heading = t.text(ctx, title, 16.5f, t.ink, true);
        header.addView(heading, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        if (subtitle != null && !subtitle.isEmpty()) {
            TextView sub = t.text(ctx, subtitle, 13f, t.inkSecondary, false);
            sub.setLineSpacing(0, 1.25f);
            sub.setPadding(0, t.dp(6), 0, t.dp(4));
            box.addView(sub);
        }
        return box;
    }

    public static TextView badge(Context ctx, StudioTheme t, String text, int bgColor, int fgColor) {
        TextView tv = t.text(ctx, text, 11f, fgColor, true);
        tv.setBackground(t.rect(bgColor, 999));
        tv.setPadding(t.dp(10), t.dp(4), t.dp(10), t.dp(4));
        return tv;
    }

    public static void styleButton(Button b, StudioTheme t, int role) {
        // role: 0 = secondary surface outline, 1 = primary accent, 2 = optical teal subtle, 3 = ghost/danger
        b.setAllCaps(false);
        b.setTextSize(14f);
        b.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        b.setGravity(Gravity.CENTER);
        b.setMinHeight(t.dp(48));
        b.setMinimumHeight(t.dp(48));
        b.setPadding(t.dp(14), t.dp(10), t.dp(14), t.dp(10));
        b.setStateListAnimator(null);

        int fill;
        int stroke;
        int normalText;
        if (role == 1) {
            fill = t.accent;
            stroke = Color.TRANSPARENT;
            normalText = t.accentInk;
        } else if (role == 2) {
            fill = t.tealSoft;
            stroke = Color.TRANSPARENT;
            normalText = t.tealSoftInk;
        } else if (role == 3) {
            fill = t.dangerSoft;
            stroke = Color.TRANSPARENT;
            normalText = t.danger;
        } else {
            fill = t.surfaceSubtle;
            stroke = t.hairline;
            normalText = t.ink;
        }
        b.setBackground(t.buttonBg(fill, stroke, 11));
        b.setTextColor(new android.content.res.ColorStateList(
                new int[][]{new int[]{-android.R.attr.state_enabled}, new int[]{}},
                new int[]{t.disabledInk, normalText}
        ));
        t.attachPressFeedback(b);
    }

    public static void renderMapResults(Context ctx, StudioTheme t, LinearLayout host,
                                        JSONObject report, TaskNameResolver resolver) {
        host.removeAllViews();
        JSONObject eval = report.has("segmentation") ? report.optJSONObject("segmentation")
                : report.optJSONObject("detection");
        if (eval == null) {
            host.setVisibility(View.GONE);
            return;
        }
        host.setVisibility(View.VISIBLE);
        boolean isSeg = "instance_segmentation".equals(eval.optString("task"));
        JSONObject metrics = eval.optJSONObject("results");
        if (metrics == null) return;

        LinearLayout card = new LinearLayout(ctx);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(t.dp(14), t.dp(12), t.dp(14), t.dp(12));
        card.setBackground(t.outlined(t.surfaceSubtle, t.hairline, 1, 12));
        host.addView(card);

        LinearLayout top = new LinearLayout(ctx);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        card.addView(top);
        top.addView(t.text(ctx, isSeg ? "Điểm tô vùng (mAP 50–95)" : "Điểm khoanh vật thể (mAP 50–95)",
                13.5f, t.ink, true), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        top.addView(badge(ctx, t, "CPU", t.tealSoft, t.tealSoftInk));

        card.addView(hairline(ctx, t, 8, 6));

        for (String key : new TreeSet<>(TestSet.TASKS)) {
            JSONObject row = metrics.optJSONObject(key);
            if (row == null) continue;
            LinearLayout line = new LinearLayout(ctx);
            line.setOrientation(LinearLayout.HORIZONTAL);
            line.setGravity(Gravity.CENTER_VERTICAL);
            line.setPadding(0, t.dp(5), 0, t.dp(5));
            card.addView(line);

            line.addView(t.text(ctx, resolver.name(key), 13f, t.inkSecondary, false),
                    new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

            if (!row.has("before_mAP50_95")) {
                line.addView(t.text(ctx, "Chưa có đáp án", 12f, t.muted, false));
            } else {
                double before = 100.0 * row.optDouble("before_mAP50_95");
                double after = 100.0 * row.optDouble("after_mAP50_95");
                double delta = after - before;
                TextView score = t.mono(ctx, String.format(Locale.US, "%.1f → %.1f", before, after), 12.5f, t.ink);
                score.setPadding(0, 0, t.dp(8), 0);
                line.addView(score);

                String deltaText = String.format(Locale.US, "%+.1f", delta);
                int bg = delta >= 0.05 ? t.tealSoft : (delta <= -0.05 ? t.dangerSoft : t.surface);
                int fg = delta >= 0.05 ? t.tealSoftInk : (delta <= -0.05 ? t.danger : t.muted);
                TextView chip = t.mono(ctx, deltaText, 11f, fg);
                chip.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
                chip.setBackground(t.rect(bg, 6));
                chip.setPadding(t.dp(7), t.dp(2), t.dp(7), t.dp(2));
                line.addView(chip);
            }
        }
    }

    public static void renderBenchmarkResults(Context ctx, StudioTheme t, LinearLayout host, JSONObject report) {
        host.removeAllViews();
        JSONObject bench = report.optJSONObject("benchmark");
        if (bench == null) {
            host.setVisibility(View.GONE);
            return;
        }
        host.setVisibility(View.VISIBLE);
        JSONObject cpu = bench.optJSONObject("cpu");
        JSONObject gpu = bench.optJSONObject("gpu_attempt");
        double cpuMs = cpu != null && cpu.has("mean_ms") ? cpu.optDouble("mean_ms", -1) : -1;
        double gpuMs = gpu != null && gpu.has("mean_ms") ? gpu.optDouble("mean_ms", -1) : -1;
        String coverage = gpu != null ? gpu.optString("coverage", "UNKNOWN") : "UNKNOWN";

        LinearLayout card = new LinearLayout(ctx);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(t.dp(14), t.dp(12), t.dp(14), t.dp(12));
        card.setBackground(t.outlined(t.surfaceSubtle, t.hairline, 1, 12));
        host.addView(card);

        LinearLayout top = new LinearLayout(ctx);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        card.addView(top);
        top.addView(t.text(ctx, "Kết quả đo 5+50 (ms / phần ảnh 256×256)", 13f, t.ink, true),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        top.addView(badge(ctx, t, coverageLabel(coverage),
                "FULL_GPU".equals(coverage) ? t.tealSoft : t.amberSoft,
                "FULL_GPU".equals(coverage) ? t.tealSoftInk : t.amber));

        double maxMs = Math.max(1.0, Math.max(cpuMs, gpuMs));
        addBarRow(ctx, t, card, "CPU (" + bench.optInt("cpu_threads", 1) + " luồng)", cpuMs, maxMs, t.inkSecondary);
        addBarRow(ctx, t, card, "GPU (LiteRT)", gpuMs, maxMs, t.teal);
    }

    private static String coverageLabel(String cov) {
        if ("FULL_GPU".equals(cov)) return "Full GPU";
        if ("PARTIAL_GPU".equals(cov)) return "GPU + CPU";
        if ("CPU_ONLY".equals(cov)) return "Chỉ CPU";
        if ("GPU_ATTEMPT_FAILED".equals(cov)) return "GPU lỗi";
        return "Chưa rõ";
    }

    private static void addBarRow(Context ctx, StudioTheme t, LinearLayout parent,
                                  String label, double ms, double maxMs, int barColor) {
        LinearLayout row = new LinearLayout(ctx);
        row.setOrientation(LinearLayout.VERTICAL);
        row.setPadding(0, t.dp(8), 0, t.dp(2));
        parent.addView(row);

        LinearLayout labels = new LinearLayout(ctx);
        labels.setOrientation(LinearLayout.HORIZONTAL);
        row.addView(labels);
        labels.addView(t.text(ctx, label, 12.5f, t.inkSecondary, false),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        String val = ms > 0 ? String.format(Locale.US, "%.2f ms", ms) : "Không khả dụng";
        TextView valTv = t.mono(ctx, val, 12.5f, ms > 0 ? t.ink : t.muted);
        valTv.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
        labels.addView(valTv);

        LinearLayout track = new LinearLayout(ctx);
        track.setBackground(t.rect(t.hairline, 4));
        LinearLayout.LayoutParams trackLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, t.dp(6));
        trackLp.setMargins(0, t.dp(5), 0, 0);
        row.addView(track, trackLp);

        if (ms > 0) {
            float fraction = (float) Math.max(0.04, Math.min(1.0, ms / maxMs));
            View fill = new View(ctx);
            fill.setBackground(t.rect(barColor, 4));
            track.addView(fill, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, fraction));
            if (fraction < 1f) {
                track.addView(new View(ctx), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f - fraction));
            }
        }
    }

    public static String shortMeta(File file) {
        if (file == null || !file.isFile() || file.length() == 0) return "Chưa có tệp trên thiết bị";
        double mb = file.length() / (1024.0 * 1024.0);
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[16384];
            try (InputStream in = new FileInputStream(file)) {
                int n = in.read(buf);
                if (n > 0) md.update(buf, 0, n);
            }
            byte[] d = md.digest();
            return String.format(Locale.US, "%.1f MB · %02x%02x%02x%02x", mb, d[0] & 255, d[1] & 255, d[2] & 255, d[3] & 255);
        } catch (Exception ignored) {
            return String.format(Locale.US, "%.1f MB", mb);
        }
    }

    public interface TaskNameResolver {
        String name(String id);
    }
}
