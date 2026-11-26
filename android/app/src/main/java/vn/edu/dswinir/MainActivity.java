package vn.edu.dswinir;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class MainActivity extends Activity {
    private static final String[] DEGRADATION_IDS = { "denoise", "derain", "dehaze", "deblur", "lowlight" };
    private static final String[] DEGRADATION_LABELS = { "Nhiễu hạt", "Vệt mưa", "Sương mù", "Ảnh nhòe", "Thiếu sáng" };
    private static final int RESTORER = 1, DETECTOR = 2, SEGMENTER = 3, IMAGE = 4, DATASET = 5, EXPORT = 6;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final List<Button> buttons = new ArrayList<>();
    private final Map<Button, BooleanSupplier> prerequisites = new HashMap<>();
    private final List<LinearLayout> pages = new ArrayList<>();
    private final List<TextView> tabs = new ArrayList<>();
    private final List<TextView> taskChips = new ArrayList<>();
    private final List<TextView> modeChips = new ArrayList<>();
    private final Handler ticker = new Handler(Looper.getMainLooper());

    private StudioTheme theme;
    private boolean busy, stoppable, isErrorState;
    private volatile boolean cancelRequested;
    private volatile Restorer activeRestorer;
    private volatile String progressSummary = "Đang xử lý";
    private int activePage = 0;
    private int selectedTaskIndex = 1; // Default to derain (matches bundled Rain100L model)
    private long jobStarted;

    private TextView readinessPill, statusText, actionHint, imageMetaLabel, zoomResetChip, datasetStatusPill;
    private TextView[] modelStatusBadges = new TextView[3];
    private TextView[] modelMetaLabels = new TextView[3];
    private View statusDot;
    private ProgressBar progress, bottomProgress;
    private ScrollView workspace;
    private LinearLayout actionBar, inspectorRow, evaluationResultsHost, benchmarkResultsHost;
    private Button restoreAction, cancelAction;
    private BeforeAfterView comparator;

    private Bitmap original, restored;
    private File restorationModel, detectionModel, segmentationModel;
    private TestSet dataset;
    private JSONObject report = new JSONObject();

    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            if (!busy)
                return;
            updateBusyHint();
            ticker.postDelayed(this, 1000);
        }
    };

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        theme = StudioTheme.of(this);
        restorationModel = new File(getFilesDir(), "restoration.tflite");
        detectionModel = new File(getFilesDir(), "detection.tflite");
        segmentationModel = new File(getFilesDir(), "segmentation.tflite");

        getWindow().setStatusBarColor(theme.bg);
        getWindow().setNavigationBarColor(theme.surface);
        applySystemBarAppearance();

        LinearLayout shell = new LinearLayout(this);
        shell.setOrientation(LinearLayout.VERTICAL);
        shell.setBackgroundColor(theme.bg);
        setContentView(shell);

        LinearLayout masthead = buildMasthead();
        shell.addView(masthead);

        LinearLayout tabStrip = buildTabs();
        shell.addView(tabStrip);
        shell.addView(StudioWidgets.hairline(this, theme, 0, 0));

        workspace = new ScrollView(this);
        workspace.setFillViewport(true);
        workspace.setOverScrollMode(View.OVER_SCROLL_IF_CONTENT_SCROLLS);
        shell.addView(workspace, new LinearLayout.LayoutParams(-1, 0, 1));

        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(theme.dp(16), theme.dp(12), theme.dp(16), theme.dp(24));
        workspace.addView(body);

        body.addView(buildStatusRibbon());

        for (int i = 0; i < 3; i++) {
            LinearLayout page = new LinearLayout(this);
            page.setOrientation(LinearLayout.VERTICAL);
            body.addView(page);
            pages.add(page);
        }

        buildRestorePage(pages.get(0));
        buildEvaluatePage(pages.get(1));
        buildSetupPage(pages.get(2));

        actionBar = buildActionBar();
        shell.addView(actionBar);

        applySystemInsets(masthead, actionBar);
        selectPage(0, false);
        refreshActions();
        refreshReportPanels();

        if (new File(getFilesDir(), "last-session/input.png").isFile()) {
            loadSession();
        }
    }

    private void applySystemInsets(View topBar, View bottomBar) {
        final int topPadLeft = topBar.getPaddingLeft();
        final int topPadTop = topBar.getPaddingTop();
        final int topPadRight = topBar.getPaddingRight();
        final int topPadBottom = topBar.getPaddingBottom();

        final int botPadLeft = bottomBar.getPaddingLeft();
        final int botPadTop = bottomBar.getPaddingTop();
        final int botPadRight = bottomBar.getPaddingRight();
        final int botPadBottom = bottomBar.getPaddingBottom();

        View root = findViewById(android.R.id.content);
        root.setOnApplyWindowInsetsListener((v, insets) -> {
            int topInset;
            int bottomInset;
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets sys = insets.getInsets(WindowInsets.Type.systemBars());
                topInset = sys.top;
                bottomInset = sys.bottom;
            } else {
                topInset = insets.getSystemWindowInsetTop();
                bottomInset = insets.getSystemWindowInsetBottom();
            }
            topBar.setPadding(topPadLeft, topPadTop + topInset, topPadRight, topPadBottom);
            bottomBar.setPadding(botPadLeft, botPadTop, botPadRight, botPadBottom + bottomInset);
            return insets;
        });
    }

    private LinearLayout buildMasthead() {
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(theme.dp(18), theme.dp(12), theme.dp(18), theme.dp(10));
        bar.setBackgroundColor(theme.bg);

        LinearLayout brandGroup = new LinearLayout(this);
        brandGroup.setOrientation(LinearLayout.HORIZONTAL);
        brandGroup.setGravity(Gravity.CENTER_VERTICAL);
        bar.addView(brandGroup, new LinearLayout.LayoutParams(0, -2, 1));

        TextView title = theme.text(this, "DSwinIR", 20f, theme.ink, true);
        title.setLetterSpacing(-0.02f);
        brandGroup.addView(title);

        TextView labTag = theme.mono(this, "STUDIO", 10f, theme.accentSoftInk);
        labTag.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
        labTag.setLetterSpacing(0.08f);
        labTag.setBackground(theme.rect(theme.accentSoft, 6));
        labTag.setPadding(theme.dp(7), theme.dp(3), theme.dp(7), theme.dp(3));
        LinearLayout.LayoutParams tagLp = new LinearLayout.LayoutParams(-2, -2);
        tagLp.setMarginStart(theme.dp(8));
        brandGroup.addView(labTag, tagLp);

        readinessPill = StudioWidgets.badge(this, theme, "Cần cài model", theme.amberSoft, theme.amber);
        readinessPill.setOnClickListener(v -> {
            if (!ready(restorationModel) && !busy)
                selectPage(2, true);
        });
        bar.addView(readinessPill);
        return bar;
    }

    private LinearLayout buildTabs() {
        LinearLayout wrap = new LinearLayout(this);
        wrap.setOrientation(LinearLayout.VERTICAL);
        wrap.setPadding(theme.dp(16), theme.dp(2), theme.dp(16), theme.dp(10));
        wrap.setBackgroundColor(theme.bg);

        LinearLayout track = new LinearLayout(this);
        track.setOrientation(LinearLayout.HORIZONTAL);
        track.setPadding(theme.dp(4), theme.dp(4), theme.dp(4), theme.dp(4));
        track.setBackground(theme.outlined(theme.surfaceSubtle, theme.hairline, 1, 12));
        wrap.addView(track, new LinearLayout.LayoutParams(-1, -2));

        String[] names = { "Khôi phục", "Đánh giá", "Thiết lập" };
        for (int i = 0; i < names.length; i++) {
            final int index = i;
            TextView tab = theme.text(this, names[i], 13f, theme.muted, true);
            tab.setGravity(Gravity.CENTER);
            tab.setMinHeight(theme.dp(38));
            tab.setPadding(theme.dp(8), theme.dp(8), theme.dp(8), theme.dp(8));
            tab.setOnClickListener(v -> selectPage(index, true));
            tabs.add(tab);
            track.addView(tab, new LinearLayout.LayoutParams(0, -2, 1));
        }
        return wrap;
    }

    private LinearLayout buildStatusRibbon() {
        LinearLayout ribbon = new LinearLayout(this);
        ribbon.setOrientation(LinearLayout.VERTICAL);
        ribbon.setPadding(theme.dp(14), theme.dp(10), theme.dp(14), theme.dp(10));
        ribbon.setBackground(theme.outlined(theme.surface, theme.hairline, 1, 12));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2);
        lp.setMargins(0, 0, 0, theme.dp(12));
        ribbon.setLayoutParams(lp);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        ribbon.addView(row);

        statusDot = new View(this);
        statusDot.setBackground(theme.rect(theme.teal, 999));
        LinearLayout.LayoutParams dotLp = new LinearLayout.LayoutParams(theme.dp(8), theme.dp(8));
        dotLp.setMarginEnd(theme.dp(10));
        row.addView(statusDot, dotLp);

        statusText = theme.text(this, "Chọn một ảnh để bắt đầu khôi phục.", 13f, theme.ink, false);
        statusText.setTextIsSelectable(true);
        statusText.setLineSpacing(0, 1.25f);
        row.addView(statusText, new LinearLayout.LayoutParams(0, -2, 1));

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setIndeterminate(true);
        progress.setIndeterminateTintList(ColorStateList.valueOf(theme.accent));
        progress.setProgressTintList(ColorStateList.valueOf(theme.accent));
        progress.setProgressBackgroundTintList(ColorStateList.valueOf(theme.hairline));
        progress.setVisibility(View.GONE);
        LinearLayout.LayoutParams pLp = new LinearLayout.LayoutParams(-1, theme.dp(4));
        pLp.setMargins(0, theme.dp(8), 0, 0);
        ribbon.addView(progress, pLp);

        return ribbon;
    }

    private void buildRestorePage(LinearLayout page) {
        // Hero interactive Before/After optical comparator
        comparator = new BeforeAfterView(this, theme);
        comparator.setOnEmptyClickListener(() -> {
            if (!busy)
                pick(IMAGE, "image/*");
        });
        comparator.setOnZoomChangedListener(() -> runOnUiThread(this::updateInspectorBar));
        LinearLayout.LayoutParams compLp = new LinearLayout.LayoutParams(-1, -2);
        compLp.setMargins(0, 0, 0, theme.dp(10));
        page.addView(comparator, compLp);

        // Optical comparator mode bar + zoom reset + dimensions
        inspectorRow = new LinearLayout(this);
        inspectorRow.setOrientation(LinearLayout.HORIZONTAL);
        inspectorRow.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams inspLp = new LinearLayout.LayoutParams(-1, -2);
        inspLp.setMargins(0, 0, 0, theme.dp(14));
        page.addView(inspectorRow, inspLp);

        LinearLayout modeGroup = new LinearLayout(this);
        modeGroup.setOrientation(LinearLayout.HORIZONTAL);
        modeGroup.setPadding(theme.dp(3), theme.dp(3), theme.dp(3), theme.dp(3));
        modeGroup.setBackground(theme.outlined(theme.surface, theme.hairline, 1, 10));
        inspectorRow.addView(modeGroup);

        String[] modeNames = { "So sánh", "Trước", "Sau" };
        BeforeAfterView.Mode[] modes = { BeforeAfterView.Mode.SPLIT, BeforeAfterView.Mode.BEFORE,
                BeforeAfterView.Mode.AFTER };
        for (int i = 0; i < modeNames.length; i++) {
            final BeforeAfterView.Mode m = modes[i];
            TextView chip = theme.text(this, modeNames[i], 12f, theme.muted, true);
            chip.setPadding(theme.dp(10), theme.dp(6), theme.dp(10), theme.dp(6));
            chip.setMinHeight(theme.dp(32));
            chip.setGravity(Gravity.CENTER);
            chip.setOnClickListener(v -> {
                comparator.setMode(m);
                updateInspectorBar();
            });
            modeChips.add(chip);
            modeGroup.addView(chip);
        }

        View spacer = new View(this);
        inspectorRow.addView(spacer, new LinearLayout.LayoutParams(0, 1, 1));

        zoomResetChip = theme.mono(this, "1.0× Đặt lại", 11.5f, theme.accentSoftInk);
        zoomResetChip.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
        zoomResetChip.setBackground(theme.rect(theme.accentSoft, 8));
        zoomResetChip.setPadding(theme.dp(9), theme.dp(5), theme.dp(9), theme.dp(5));
        zoomResetChip.setVisibility(View.GONE);
        zoomResetChip.setOnClickListener(v -> comparator.resetZoom());
        LinearLayout.LayoutParams zrLp = new LinearLayout.LayoutParams(-2, -2);
        zrLp.setMarginEnd(theme.dp(8));
        inspectorRow.addView(zoomResetChip, zrLp);

        imageMetaLabel = theme.mono(this, "Chưa có ảnh", 11.5f, theme.muted);
        inspectorRow.addView(imageMetaLabel);

        // Section 01: Image source & degradation selector
        LinearLayout photoSection = StudioWidgets.sectionBlock(this, theme, page,
                "01", "Nguồn ảnh & suy thoái", null);
        addButton(photoSection, "Chọn ảnh từ thiết bị", 0, () -> pick(IMAGE, "image/*"), () -> true);

        TextView taskLabel = theme.text(this, "Loại suy thoái cần ghi nhận", 12f, theme.muted, true);
        taskLabel.setPadding(0, theme.dp(10), 0, theme.dp(6));
        photoSection.addView(taskLabel);

        HorizontalScrollView chipScroll = new HorizontalScrollView(this);
        chipScroll.setHorizontalScrollBarEnabled(false);
        photoSection.addView(chipScroll, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout chipRow = new LinearLayout(this);
        chipRow.setOrientation(LinearLayout.HORIZONTAL);
        chipScroll.addView(chipRow);

        for (int i = 0; i < DEGRADATION_LABELS.length; i++) {
            final int idx = i;
            TextView chip = theme.text(this, DEGRADATION_LABELS[i], 12.5f, theme.inkSecondary, true);
            chip.setMinHeight(theme.dp(36));
            chip.setGravity(Gravity.CENTER);
            chip.setPadding(theme.dp(13), theme.dp(7), theme.dp(13), theme.dp(7));
            chip.setOnClickListener(v -> {
                if (busy)
                    return;
                selectedTaskIndex = idx;
                refreshTaskChips();
            });
            LinearLayout.LayoutParams cLp = new LinearLayout.LayoutParams(-2, -2);
            cLp.setMarginEnd(theme.dp(8));
            chipRow.addView(chip, cLp);
            taskChips.add(chip);
        }

        // Section 02: GPU & Export
        LinearLayout extraSection = StudioWidgets.sectionBlock(this, theme, page,
                "02", "Tăng tốc & lưu kết quả",
                "Thử chạy trên GPU hoặc lưu ảnh gốc, ảnh khôi phục và báo cáo JSON thành tệp .zip.");
        addButton(extraSection, "Thử khôi phục bằng bộ xử lý đồ họa", 2, () -> restore(true), this::canRestore);
        addButton(extraSection, "Lưu ảnh và kết quả", 0, this::export, () -> restored != null || report.length() > 0);
    }

    private void buildEvaluatePage(LinearLayout page) {
        LinearLayout visionSection = StudioWidgets.sectionBlock(this, theme, page,
                "01", "Nhận diện vật thể (YOLO11)",
                "So sánh khả năng phát hiện vật thể trên ảnh gốc và ảnh sau khôi phục.");
        addButton(visionSection, "Tìm và khoanh vật thể", 2, () -> preview(false),
                () -> restored != null && ready(detectionModel));
        addButton(visionSection, "Tô vùng từng vật thể", 2, () -> preview(true),
                () -> restored != null && ready(segmentationModel));

        LinearLayout batchSection = StudioWidgets.sectionBlock(this, theme, page,
                "02", "Bộ ảnh kiểm tra (.zip) & mAP",
                "Chạy hàng loạt trên tệp ZIP. Chấm điểm mAP yêu cầu bộ ảnh đã có nhãn vị trí hoặc vùng vật thể.");
        datasetStatusPill = StudioWidgets.badge(this, theme, "Chưa chọn bộ ảnh", theme.surfaceSubtle, theme.muted);
        LinearLayout.LayoutParams dsLp = new LinearLayout.LayoutParams(-2, -2);
        dsLp.setMargins(0, theme.dp(4), 0, theme.dp(8));
        batchSection.addView(datasetStatusPill, dsLp);

        addButton(batchSection, "Chọn bộ ảnh kiểm tra (.zip)", 0, () -> pick(DATASET, "*/*"), () -> true);
        addButton(batchSection, "Khôi phục tất cả ảnh trong bộ", 0, this::restoreBatch,
                () -> dataset != null && ready(restorationModel));
        addButton(batchSection, "Chấm điểm tìm vật thể", 2, () -> evaluate(false),
                () -> ready(restorationModel) && ready(detectionModel) && dataset != null
                        && dataset.hasGroundTruth(false));
        addButton(batchSection, "Chấm điểm tô vùng vật thể", 2, () -> evaluate(true),
                () -> ready(restorationModel) && ready(segmentationModel) && dataset != null
                        && dataset.hasGroundTruth(true));

        evaluationResultsHost = new LinearLayout(this);
        evaluationResultsHost.setOrientation(LinearLayout.VERTICAL);
        evaluationResultsHost.setPadding(0, theme.dp(10), 0, 0);
        evaluationResultsHost.setVisibility(View.GONE);
        batchSection.addView(evaluationResultsHost);

        LinearLayout benchSection = StudioWidgets.sectionBlock(this, theme, page,
                "03", "Đo tốc độ 5+50 (Benchmark)",
                "Chạy thử 5 lần rồi đo 50 lần trên tile 256×256 để so sánh CPU và GPU. Trên máy ảo có thể mất nhiều phút.");
        addButton(benchSection, "Bắt đầu đo tốc độ", 0,
                () -> new AlertDialog.Builder(this)
                        .setTitle("Đo tốc độ chuyên sâu?")
                        .setMessage(
                                "Phép đo chạy thử 5 lần rồi đo 50 lần cho CPU và GPU. Trên máy ảo có thể mất nhiều phút; bạn có thể bấm Dừng xử lý bất kỳ lúc nào.")
                        .setNegativeButton("Để sau", null)
                        .setPositiveButton("Đo tốc độ", (d, w) -> benchmark())
                        .show(),
                this::canRestore);

        benchmarkResultsHost = new LinearLayout(this);
        benchmarkResultsHost.setOrientation(LinearLayout.VERTICAL);
        benchmarkResultsHost.setPadding(0, theme.dp(10), 0, 0);
        benchmarkResultsHost.setVisibility(View.GONE);
        benchSection.addView(benchmarkResultsHost);
    }

    private void buildSetupPage(LinearLayout page) {
        LinearLayout modelsSection = StudioWidgets.sectionBlock(this, theme, page,
                "01", "Cài công cụ mô hình (.tflite)",
                "Chọn tệp TFLite FP32 tương ứng từ bộ nhớ thiết bị. Chỉ cần cài một lần cho mỗi công cụ.");
        modelsSection.addView(StudioWidgets.hairline(this, theme, 6, 10));

        addModelRow(modelsSection, 0, "Khôi phục ảnh · DSwinIR", "Yêu cầu đầu vào 256×256 FP32",
                "Cài công cụ khôi phục ảnh", () -> pick(RESTORER, "*/*"));
        modelsSection.addView(StudioWidgets.hairline(this, theme, 10, 10));
        addModelRow(modelsSection, 1, "Khoanh vật thể · YOLO11", "COCO 80 lớp, nms=False",
                "Cài công cụ khoanh vật thể", () -> pick(DETECTOR, "*/*"));
        modelsSection.addView(StudioWidgets.hairline(this, theme, 10, 10));
        addModelRow(modelsSection, 2, "Tô vùng vật thể · YOLO11-Seg", "COCO 80 lớp kèm prototype mask",
                "Cài công cụ tô vùng vật thể", () -> pick(SEGMENTER, "*/*"));

        LinearLayout themeSection = StudioWidgets.sectionBlock(this, theme, page,
                "02", "Giao diện hiển thị",
                "Chọn chế độ sáng, tối hoặc tự động theo cài đặt hệ thống.");
        addThemeSelector(themeSection);

        LinearLayout notice = new LinearLayout(this);
        notice.setOrientation(LinearLayout.VERTICAL);
        notice.setPadding(theme.dp(14), theme.dp(12), theme.dp(14), theme.dp(12));
        notice.setBackground(theme.outlined(theme.amberSoft, theme.amber, 1, 12));
        LinearLayout.LayoutParams nLp = new LinearLayout.LayoutParams(-1, -2);
        nLp.setMargins(0, 0, 0, theme.dp(12));
        page.addView(notice, nLp);

        TextView nt = theme.text(this, "Lưu ý phạm vi mô hình", 13f, theme.amber, true);
        notice.addView(nt);
        TextView nb = theme.text(this,
                "Công cụ khôi phục đi kèm được huấn luyện chuyên cho tác vụ xóa vệt mưa (Rain100L). Kết quả trên ảnh nhiễu, sương mù, nhòe hoặc thiếu sáng chỉ mang tính thử nghiệm ngoài phân phối.",
                12.5f, theme.inkSecondary, false);
        nb.setLineSpacing(0, 1.3f);
        nb.setPadding(0, theme.dp(4), 0, 0);
        notice.addView(nb);
    }

    private void applySystemBarAppearance() {
        if (Build.VERSION.SDK_INT >= 30) {
            android.view.WindowInsetsController ctrl = getWindow().getInsetsController();
            if (ctrl != null) {
                int lightMask = android.view.WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS
                        | android.view.WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS;
                ctrl.setSystemBarsAppearance(theme.dark ? 0 : lightMask, lightMask);
            }
        } else {
            @SuppressWarnings("deprecation")
            int flags = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR;
            @SuppressWarnings("deprecation")
            int current = getWindow().getDecorView().getSystemUiVisibility();
            @SuppressWarnings("deprecation")
            int next = theme.dark ? (current & ~flags) : (current | flags);
            getWindow().getDecorView().setSystemUiVisibility(next);
        }
    }

    private void addThemeSelector(LinearLayout parent) {
        String current = getSharedPreferences(StudioTheme.PREFS, MODE_PRIVATE)
                .getString(StudioTheme.KEY_THEME_MODE, "system");
        LinearLayout track = new LinearLayout(this);
        track.setOrientation(LinearLayout.HORIZONTAL);
        track.setPadding(theme.dp(4), theme.dp(4), theme.dp(4), theme.dp(4));
        track.setBackground(theme.outlined(theme.surfaceSubtle, theme.hairline, 1, 12));
        LinearLayout.LayoutParams tLp = new LinearLayout.LayoutParams(-1, -2);
        tLp.setMargins(0, theme.dp(6), 0, 0);
        parent.addView(track, tLp);

        String[] keys = { "system", "light", "dark" };
        String[] labels = { "Hệ thống", "Sáng", "Tối" };
        for (int i = 0; i < keys.length; i++) {
            final String key = keys[i];
            boolean selected = key.equals(current);
            TextView pill = theme.text(this, labels[i], 13f, selected ? theme.ink : theme.muted, true);
            pill.setGravity(Gravity.CENTER);
            pill.setMinHeight(theme.dp(38));
            pill.setPadding(theme.dp(8), theme.dp(8), theme.dp(8), theme.dp(8));
            if (selected)
                pill.setBackground(theme.outlined(theme.surface, theme.hairline, 1, 9));
            pill.setOnClickListener(v -> {
                if (busy || key.equals(current))
                    return;
                getSharedPreferences(StudioTheme.PREFS, MODE_PRIVATE).edit()
                        .putString(StudioTheme.KEY_THEME_MODE, key).apply();
                recreate();
            });
            track.addView(pill, new LinearLayout.LayoutParams(0, -2, 1));
        }
    }

    private void addModelRow(LinearLayout parent, int slot, String name, String spec,
            String buttonLabel, Runnable action) {
        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        parent.addView(top);

        LinearLayout info = new LinearLayout(this);
        info.setOrientation(LinearLayout.VERTICAL);
        top.addView(info, new LinearLayout.LayoutParams(0, -2, 1));

        info.addView(theme.text(this, name, 14f, theme.ink, true));
        TextView specTv = theme.text(this, spec, 12f, theme.muted, false);
        specTv.setPadding(0, theme.dp(2), 0, 0);
        info.addView(specTv);

        TextView badge = StudioWidgets.badge(this, theme, "Chưa cài", theme.surfaceSubtle, theme.muted);
        modelStatusBadges[slot] = badge;
        top.addView(badge);

        TextView meta = theme.text(this, "Chưa có tệp trên thiết bị", 12f, theme.muted, false);
        meta.setFontFeatureSettings("tnum");
        meta.setPadding(0, theme.dp(6), 0, theme.dp(4));
        modelMetaLabels[slot] = meta;
        parent.addView(meta);

        addButton(parent, buttonLabel, 0, action, () -> true);
    }

    private LinearLayout buildActionBar() {
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.VERTICAL);
        bar.setBackgroundColor(theme.surface);
        bar.setPadding(theme.dp(16), theme.dp(10), theme.dp(16), theme.dp(12));

        bar.addView(StudioWidgets.hairline(this, theme, 0, 6));

        bottomProgress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bottomProgress.setIndeterminate(true);
        bottomProgress.setIndeterminateTintList(ColorStateList.valueOf(theme.accent));
        bottomProgress.setProgressTintList(ColorStateList.valueOf(theme.accent));
        bottomProgress.setProgressBackgroundTintList(ColorStateList.valueOf(theme.hairline));
        bottomProgress.setVisibility(View.GONE);
        LinearLayout.LayoutParams bpLp = new LinearLayout.LayoutParams(-1, theme.dp(3));
        bpLp.setMargins(0, 0, 0, theme.dp(6));
        bar.addView(bottomProgress, bpLp);

        actionHint = theme.text(this, "Chọn ảnh để bắt đầu.", 12.5f, theme.muted, false);
        actionHint.setMaxLines(1);
        actionHint.setEllipsize(TextUtils.TruncateAt.END);
        actionHint.setPadding(0, 0, 0, theme.dp(6));
        bar.addView(actionHint);

        restoreAction = new Button(this);
        restoreAction.setText("Bắt đầu khôi phục");
        StudioWidgets.styleButton(restoreAction, theme, 1);
        restoreAction.setOnClickListener(v -> restore(false));
        buttons.add(restoreAction);
        prerequisites.put(restoreAction, this::canRestore);
        bar.addView(restoreAction, new LinearLayout.LayoutParams(-1, -2));

        cancelAction = new Button(this);
        cancelAction.setText("Dừng xử lý");
        StudioWidgets.styleButton(cancelAction, theme, 3);
        cancelAction.setOnClickListener(v -> cancelJob());
        LinearLayout.LayoutParams cLp = new LinearLayout.LayoutParams(-1, -2);
        cLp.setMargins(0, theme.dp(8), 0, 0);
        bar.addView(cancelAction, cLp);

        return bar;
    }

    private Button addButton(LinearLayout parent, String label, int role,
            Runnable action, BooleanSupplier available) {
        Button b = new Button(this);
        b.setText(label);
        StudioWidgets.styleButton(b, theme, role);
        b.setOnClickListener(v -> action.run());
        buttons.add(b);
        prerequisites.put(b, available);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2);
        lp.setMargins(0, theme.dp(6), 0, theme.dp(2));
        parent.addView(b, lp);
        return b;
    }

    private void selectPage(int index, boolean animate) {
        activePage = index;
        for (int i = 0; i < pages.size(); i++) {
            LinearLayout p = pages.get(i);
            if (i == index) {
                p.setVisibility(View.VISIBLE);
                if (animate && !theme.reducedMotion) {
                    p.setAlpha(0f);
                    p.setTranslationY(theme.dpf(8));
                    p.animate().alpha(1f).translationY(0f).setDuration(220)
                            .setInterpolator(StudioTheme.EASE_OUT_QUINT).start();
                } else {
                    p.setAlpha(1f);
                    p.setTranslationY(0f);
                }
            } else {
                p.setVisibility(View.GONE);
            }
        }
        for (int i = 0; i < tabs.size(); i++) {
            TextView tab = tabs.get(i);
            boolean selected = i == index;
            tab.setBackground(selected ? theme.outlined(theme.surface, theme.hairline, 1, 9) : null);
            tab.setTextColor(selected ? theme.ink : theme.muted);
        }
        if (actionBar != null) {
            actionBar.setVisibility(index == 0 || busy ? View.VISIBLE : View.GONE);
        }
    }

    private void refreshTaskChips() {
        for (int i = 0; i < taskChips.size(); i++) {
            TextView chip = taskChips.get(i);
            boolean selected = i == selectedTaskIndex;
            chip.setBackground(selected
                    ? theme.outlined(theme.ink, Color.TRANSPARENT, 0, 999)
                    : theme.outlined(theme.surfaceSubtle, theme.hairline, 1, 999));
            chip.setTextColor(selected ? theme.surface : theme.inkSecondary);
            chip.setEnabled(!busy);
            chip.setAlpha(busy ? 0.55f : 1f);
        }
    }

    private void updateInspectorBar() {
        if (inspectorRow == null)
            return;
        boolean hasOriginal = original != null;
        boolean hasRestored = restored != null;
        BeforeAfterView.Mode current = comparator.getMode();
        for (int i = 0; i < modeChips.size(); i++) {
            TextView chip = modeChips.get(i);
            boolean available = (i == 1 && hasOriginal) || ((i == 0 || i == 2) && hasRestored);
            boolean selected = (i == 0 && current == BeforeAfterView.Mode.SPLIT)
                    || (i == 1 && current == BeforeAfterView.Mode.BEFORE)
                    || (i == 2 && current == BeforeAfterView.Mode.AFTER);
            chip.setEnabled(available);
            chip.setBackground(selected && available ? theme.rect(theme.ink, 7) : null);
            chip.setTextColor(!available ? theme.disabledInk : (selected ? theme.surface : theme.inkSecondary));
        }

        float z = comparator.getZoom();
        zoomResetChip.setVisibility(hasOriginal && z > 1.05f ? View.VISIBLE : View.GONE);
        if (hasOriginal) {
            zoomResetChip.setText(String.format(Locale.US, "%.1f× Đặt lại", z));
            imageMetaLabel.setText(original.getWidth() + "×" + original.getHeight());
        } else {
            imageMetaLabel.setText("Chưa có ảnh");
        }
    }

    private void refreshModelInventory() {
        File[] files = { restorationModel, detectionModel, segmentationModel };
        for (int i = 0; i < files.length; i++) {
            boolean ok = ready(files[i]);
            TextView badge = modelStatusBadges[i];
            badge.setText(ok ? "Đã cài" : "Chưa cài");
            badge.setBackground(theme.rect(ok ? theme.tealSoft : theme.surfaceSubtle, 999));
            badge.setTextColor(ok ? theme.tealSoftInk : theme.muted);
            modelMetaLabels[i].setText(StudioWidgets.shortMeta(files[i]));
        }
    }

    private void refreshReportPanels() {
        if (evaluationResultsHost != null) {
            StudioWidgets.renderMapResults(this, theme, evaluationResultsHost, report, this::taskName);
        }
        if (benchmarkResultsHost != null) {
            StudioWidgets.renderBenchmarkResults(this, theme, benchmarkResultsHost, report);
        }
    }

    private static boolean ready(File file) {
        return file != null && file.isFile() && file.length() > 0;
    }

    private boolean canRestore() {
        return original != null && ready(restorationModel);
    }

    private void refreshActions() {
        for (Button b : buttons) {
            BooleanSupplier check = prerequisites.get(b);
            b.setEnabled(!busy && check != null && check.getAsBoolean());
        }
        refreshTaskChips();
        updateInspectorBar();
        refreshModelInventory();

        progress.setVisibility(busy ? View.VISIBLE : View.GONE);
        bottomProgress.setVisibility(busy ? View.VISIBLE : View.GONE);
        cancelAction.setVisibility(busy && stoppable ? View.VISIBLE : View.GONE);
        cancelAction.setEnabled(!cancelRequested);
        restoreAction.setText(busy ? "Đang xử lý ảnh…" : "Bắt đầu khôi phục");

        if (!busy) {
            if (restored != null) {
                actionHint.setText("Đã có ảnh kết quả · Kéo thanh trượt trên ảnh để so sánh.");
            } else if (original == null) {
                actionHint.setText("Bước 1: Chọn ảnh từ thiết bị để bắt đầu.");
            } else if (ready(restorationModel)) {
                actionHint.setText(
                        "Ảnh " + original.getWidth() + "×" + original.getHeight() + " đã sẵn sàng để khôi phục.");
            } else {
                actionHint.setText("Mở mục Thiết lập để cài công cụ khôi phục ảnh.");
            }
        }
        actionBar.setVisibility(activePage == 0 || busy ? View.VISIBLE : View.GONE);

        // Top readiness pill & status dot
        if (busy) {
            readinessPill.setText("Đang chạy");
            readinessPill.setBackground(theme.rect(theme.accentSoft, 999));
            readinessPill.setTextColor(theme.accentSoftInk);
            statusDot.setBackground(theme.rect(theme.accent, 999));
        } else if (isErrorState) {
            readinessPill.setText(ready(restorationModel) ? "Sẵn sàng" : "Cần cài model");
            readinessPill.setBackground(theme.rect(theme.dangerSoft, 999));
            readinessPill.setTextColor(theme.danger);
            statusDot.setBackground(theme.rect(theme.danger, 999));
        } else if (!ready(restorationModel)) {
            readinessPill.setText("Cần cài model");
            readinessPill.setBackground(theme.rect(theme.amberSoft, 999));
            readinessPill.setTextColor(theme.amber);
            statusDot.setBackground(theme.rect(theme.amber, 999));
        } else {
            readinessPill.setText("Sẵn sàng");
            readinessPill.setBackground(theme.rect(theme.tealSoft, 999));
            readinessPill.setTextColor(theme.tealSoftInk);
            statusDot.setBackground(theme.rect(theme.teal, 999));
        }

        if (datasetStatusPill != null) {
            if (dataset == null) {
                datasetStatusPill.setText("Chưa chọn bộ ảnh");
                datasetStatusPill.setBackground(theme.rect(theme.surfaceSubtle, 999));
                datasetStatusPill.setTextColor(theme.muted);
            } else if (dataset.hasGroundTruth(true)) {
                datasetStatusPill.setText(dataset.images.length() + " ảnh · Có đáp án bbox & vùng");
                datasetStatusPill.setBackground(theme.rect(theme.tealSoft, 999));
                datasetStatusPill.setTextColor(theme.tealSoftInk);
            } else if (dataset.hasGroundTruth(false)) {
                datasetStatusPill.setText(dataset.images.length() + " ảnh · Có đáp án bbox");
                datasetStatusPill.setBackground(theme.rect(theme.tealSoft, 999));
                datasetStatusPill.setTextColor(theme.tealSoftInk);
            } else {
                datasetStatusPill.setText(dataset.images.length() + " ảnh · Chưa có nhãn mAP");
                datasetStatusPill.setBackground(theme.rect(theme.amberSoft, 999));
                datasetStatusPill.setTextColor(theme.amber);
            }
        }
    }

    private String taskName(String id) {
        switch (id) {
            case "denoise":
                return "Ảnh nhiễu";
            case "derain":
                return "Ảnh có mưa";
            case "dehaze":
                return "Ảnh có sương";
            case "deblur":
                return "Ảnh nhòe";
            case "lowlight":
                return "Ảnh thiếu sáng";
            default:
                return id;
        }
    }

    private void updateBusyHint() {
        long sec = (SystemClock.elapsedRealtime() - jobStarted) / 1000;
        String prefix = cancelRequested ? "Đang dừng · " : "";
        actionHint.setText(prefix + progressSummary + " · " + sec + " giây");
    }

    private void show(String text) {
        isErrorState = false;
        runOnUiThread(() -> {
            statusText.setText(text);
            statusDot.setBackground(theme.rect(busy ? theme.accent : theme.teal, 999));
        });
    }

    private void showProgress(String summary, String detailText) {
        progressSummary = summary;
        isErrorState = false;
        runOnUiThread(() -> {
            statusText.setText(detailText);
            statusDot.setBackground(theme.rect(theme.accent, 999));
            if (busy)
                updateBusyHint();
        });
    }

    private void showError(String text) {
        isErrorState = true;
        runOnUiThread(() -> {
            statusText.setText(text);
            statusDot.setBackground(theme.rect(theme.danger, 999));
        });
    }

    private void setProgressValue(int done, int total) {
        runOnUiThread(() -> {
            ProgressBar[] bars = { progress, bottomProgress };
            for (ProgressBar bar : bars) {
                if (bar == null)
                    continue;
                if (total <= 0) {
                    bar.setIndeterminate(true);
                } else {
                    bar.setIndeterminate(false);
                    bar.setMax(total);
                    bar.setProgress(Math.max(0, Math.min(total, done)));
                }
            }
        });
    }

    private interface Job {
        void run() throws Exception;
    }

    private void checkCancelled() {
        if (cancelRequested)
            throw new CancellationException();
    }

    private Restorer openRestorer(File file, boolean gpu) {
        checkCancelled();
        Restorer engine = new Restorer(file, gpu);
        activeRestorer = engine;
        if (cancelRequested) {
            engine.cancel();
            engine.close();
            throw new CancellationException();
        }
        return engine;
    }

    private void cancelJob() {
        cancelRequested = true;
        cancelAction.setEnabled(false);
        show("Đang dừng xử lý. Chờ phần tính toán hiện tại hoàn tất…");
        updateBusyHint();
        Restorer engine = activeRestorer;
        if (engine != null)
            engine.cancel();
    }

    private void job(Job action) {
        job(false, action);
    }

    private void job(boolean allowStop, Job action) {
        if (busy)
            return;
        cancelRequested = false;
        activeRestorer = null;
        stoppable = allowStop;
        busy = true;
        isErrorState = false;
        progressSummary = "Đang xử lý";
        jobStarted = SystemClock.elapsedRealtime();
        progress.setIndeterminate(true);
        if (bottomProgress != null)
            bottomProgress.setIndeterminate(true);
        refreshActions();
        ticker.removeCallbacks(tick);
        ticker.post(tick);
        if (activePage == 0) {
            workspace.smoothScrollTo(0, 0);
        }
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        worker.execute(() -> {
            try {
                action.run();
            } catch (CancellationException e) {
                show("Đã dừng xử lý. Kết quả trước đó vẫn được giữ nguyên.");
            } catch (Exception | LinkageError e) {
                showError("Không thực hiện được: " + e.getMessage());
            } finally {
                activeRestorer = null;
                runOnUiThread(() -> {
                    busy = false;
                    ticker.removeCallbacks(tick);
                    getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                    refreshActions();
                    refreshReportPanels();
                });
            }
        });
    }

    private void pick(int code, String mime) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT)
                .setType(mime)
                .addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, code);
    }

    @Override
    protected void onActivityResult(int code, int result, Intent data) {
        super.onActivityResult(code, result, data);
        if (result != RESULT_OK || data == null || data.getData() == null)
            return;
        Uri uri = data.getData();
        job(() -> {
            if (code == IMAGE) {
                Bitmap decoded = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), uri),
                        (d, info, s) -> {
                            if ((long) info.getSize().getWidth() * info.getSize().getHeight() > 24000000) {
                                throw new IllegalArgumentException(
                                        "Ảnh quá lớn. Hãy chọn ảnh nhỏ hơn 24 triệu điểm ảnh.");
                            }
                            d.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                        });
                if (decoded.getWidth() * (long) decoded.getHeight() > 24000000) {
                    decoded.recycle();
                    throw new IOException("Ảnh quá lớn. Hãy chọn ảnh nhỏ hơn 24 triệu điểm ảnh.");
                }
                original = decoded.copy(Bitmap.Config.ARGB_8888, false);
                decoded.recycle();
                restored = null;
                runOnUiThread(() -> {
                    comparator.setImages(original, null);
                    selectPage(0, false);
                });
                report = new JSONObject();
                saveSession();
                show("Đã mở ảnh " + original.getWidth() + "×" + original.getHeight()
                        + ". Nhấn «Bắt đầu khôi phục» bên dưới.");
            } else if (code == DATASET) {
                File folder = new File(getCacheDir(), "dataset-" + System.currentTimeMillis());
                dataset = TestSet.unpack(getContentResolver().openInputStream(uri), folder);
                report.remove("detection");
                report.remove("segmentation");
                report.remove("batch_restoration");
                show("Đã nhập " + dataset.images.length() + " ảnh kiểm tra."
                        + (dataset.hasGroundTruth(false) ? " Bộ ảnh có nhãn để chấm điểm mAP."
                                : " Chưa có nhãn đáp án nên chỉ hỗ trợ khôi phục hàng loạt."));
            } else if (code == EXPORT) {
                try (OutputStream output = getContentResolver().openOutputStream(uri);
                        ZipOutputStream zip = new ZipOutputStream(output)) {
                    zip.putNextEntry(new ZipEntry("report.json"));
                    zip.write(report.toString(2).getBytes(StandardCharsets.UTF_8));
                    zip.closeEntry();
                    if (original != null) {
                        zip.putNextEntry(new ZipEntry("degraded.png"));
                        original.compress(Bitmap.CompressFormat.PNG, 100, zip);
                        zip.closeEntry();
                    }
                    if (restored != null) {
                        zip.putNextEntry(new ZipEntry("restored.png"));
                        restored.compress(Bitmap.CompressFormat.PNG, 100, zip);
                        zip.closeEntry();
                    }
                    File images = new File(getFilesDir(), "evaluation-images");
                    if ((report.has("detection") || report.has("segmentation") || report.has("batch_restoration"))
                            && images.isDirectory()) {
                        for (File f : Objects.requireNonNull(images.listFiles())) {
                            if (f.getName().startsWith("."))
                                continue;
                            zip.putNextEntry(new ZipEntry("evaluation/" + f.getName()));
                            java.nio.file.Files.copy(f.toPath(), zip);
                            zip.closeEntry();
                        }
                    }
                }
                show("Đã lưu ảnh và báo cáo vào tệp nén .zip.");
            } else {
                File target = code == RESTORER ? restorationModel
                        : code == DETECTOR ? detectionModel : segmentationModel;
                File pending = new File(getCacheDir(), "import.tflite");
                try (InputStream in = getContentResolver().openInputStream(uri);
                        OutputStream out = new FileOutputStream(pending)) {
                    copy(in, out, 1024L * 1024 * 1024);
                }
                if (code == RESTORER) {
                    try (Restorer check = new Restorer(pending, false)) {
                        /* validate IO */ }
                } else {
                    try (Yolo check = new Yolo(pending, code == SEGMENTER)) {
                        /* validate task */ }
                }
                java.nio.file.Files.copy(pending.toPath(), target.toPath(),
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                pending.delete();
                if (code == RESTORER) {
                    restored = null;
                    report = new JSONObject();
                    saveSession();
                    runOnUiThread(() -> comparator.setImages(original, null));
                }
                show("Đã cài xong công cụ. " + (original == null ? "Chuyển sang Khôi phục để chọn ảnh."
                        : "Nhấn «Bắt đầu khôi phục» để chạy."));
            }
        });
    }

    private void saveSession() {
        if (original == null)
            return;
        try {
            File dir = new File(getFilesDir(), "last-session");
            dir.mkdirs();
            new File(dir, "output.png").delete();
            new File(dir, "report.json").delete();
            saveBitmap(original, new File(dir, "input.png"));
            if (restored != null)
                saveBitmap(restored, new File(dir, "output.png"));
            else
                new File(dir, "output.png").delete();
            java.nio.file.Files.write(new File(dir, "report.json").toPath(),
                    report.toString().getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            android.util.Log.w("DSwinIR", "Cannot save last session", e);
        }
    }

    private void saveBitmap(Bitmap image, File file) throws IOException {
        File pending = new File(file.getPath() + ".pending");
        try (OutputStream out = new FileOutputStream(pending)) {
            if (!image.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                throw new IOException("Cannot save image");
            }
        }
        java.nio.file.Files.move(pending.toPath(), file.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }

    private void loadSession() {
        job(() -> {
            File dir = new File(getFilesDir(), "last-session");
            original = BitmapFactory.decodeFile(new File(dir, "input.png").getPath());
            restored = BitmapFactory.decodeFile(new File(dir, "output.png").getPath());
            if (original == null)
                throw new IOException("Không mở được ảnh đã lưu. Hãy chọn lại ảnh.");
            if (restored != null
                    && (restored.getWidth() != original.getWidth() || restored.getHeight() != original.getHeight())) {
                restored = null;
            }
            File saved = new File(dir, "report.json");
            if (saved.isFile()) {
                report = new JSONObject(
                        new String(java.nio.file.Files.readAllBytes(saved.toPath()), StandardCharsets.UTF_8));
            }
            runOnUiThread(() -> comparator.setImages(original, restored));
            show(restored == null
                    ? "Đã mở lại ảnh vừa chọn. Nhấn «Bắt đầu khôi phục» để tiếp tục."
                    : "Đã mở lại phiên trước. Kéo thanh trượt trên ảnh để so sánh Trước và Sau.");
        });
    }

    private static String sha(File file) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] b = new byte[65536];
        try (InputStream in = new FileInputStream(file)) {
            int n;
            while ((n = in.read(b)) != -1)
                md.update(b, 0, n);
        }
        StringBuilder s = new StringBuilder();
        for (byte v : md.digest())
            s.append(String.format(Locale.US, "%02x", v & 255));
        return s.toString();
    }

    private static void copy(InputStream in, OutputStream out, long limit) throws IOException {
        byte[] buffer = new byte[65536];
        long total = 0;
        int n;
        while ((n = in.read(buffer)) != -1) {
            total += n;
            if (total > limit)
                throw new IOException("File exceeds size limit");
            out.write(buffer, 0, n);
        }
    }

    private void requireImage() {
        if (original == null)
            throw new IllegalStateException("Hãy chọn một ảnh trước.");
    }

    private void restore(boolean gpu) {
        final String degradation = DEGRADATION_IDS[selectedTaskIndex];
        job(true, () -> {
            requireImage();
            show("Đang khởi tạo bộ khôi phục" + (gpu ? " GPU…" : " CPU…"));
            String requested = gpu ? "GPU" : "CPU", error = null;
            long start = SystemClock.elapsedRealtimeNanos();
            Restorer.Progress onProgress = (done, total) -> {
                setProgressValue(done, total);
                String summary = "Tile " + (done == total ? done : (done + 1)) + "/" + total;
                showProgress(summary,
                        "Đang khôi phục · " + (done == total ? "hoàn tất " + done : "đang xử lý " + (done + 1))
                                + "/" + total + " phần ảnh (256×256)");
            };
            try (Restorer engine = openRestorer(restorationModel, gpu)) {
                restored = engine.restore(original, onProgress);
            } catch (RuntimeException | Error e) {
                if (e instanceof ThreadDeath)
                    throw e;
                checkCancelled();
                if (e instanceof CancellationException)
                    throw e;
                if (!gpu)
                    throw e;
                error = e.toString();
                requested = "CPU_FALLBACK";
                show("GPU không hỗ trợ, tự động chuyển sang CPU…");
                System.gc();
                try (Restorer engine = openRestorer(restorationModel, false)) {
                    restored = engine.restore(original, onProgress);
                }
            }
            JSONObject r = new JSONObject()
                    .put("requested", gpu ? "GPU" : "CPU")
                    .put("executed", requested)
                    .put("gpuCoverage", gpu ? "UNKNOWN_GPU_COVERAGE" : "CPU_ONLY")
                    .put("degradation", degradation)
                    .put("model_sha256", sha(restorationModel))
                    .put("width", restored.getWidth())
                    .put("height", restored.getHeight())
                    .put("end_to_end_ms", (SystemClock.elapsedRealtimeNanos() - start) / 1e6)
                    .put("gpu_error", error);
            report.put("restoration", r);
            saveSession();
            runOnUiThread(() -> {
                comparator.setImages(original, restored);
                selectPage(0, false);
                workspace.smoothScrollTo(0, 0);
            });
            show("Hoàn tất khôi phục " + restored.getWidth() + "×" + restored.getHeight()
                    + String.format(Locale.US, " trong %.1f giây (%s).", r.getDouble("end_to_end_ms") / 1000,
                            requested));
        });
    }

    private String readGpuLogs(String since) {
        File log = new File(getCacheDir(), "gpu-runtime.log");
        try {
            java.lang.Process p = new ProcessBuilder("logcat", "-d", "-v", "raw",
                    "--pid=" + android.os.Process.myPid(), "-T", since, "tflite:V", "*:S")
                    .redirectErrorStream(true).redirectOutput(log).start();
            if (!p.waitFor(3, TimeUnit.SECONDS)) {
                p.destroy();
                return "Logcat unavailable";
            }
            return new String(java.nio.file.Files.readAllBytes(log.toPath()), StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "Logcat unavailable: " + e;
        }
    }

    private JSONObject measure(boolean gpu) throws Exception {
        String since = new SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US).format(new Date());
        JSONObject value = new JSONObject()
                .put("requested", gpu ? "GPU" : "CPU")
                .put("warmup", 5)
                .put("runs", 50)
                .put("unit", "ms/tile 256x256; interpreter.run including output readback");
        try (Restorer engine = openRestorer(restorationModel, gpu)) {
            double[] samples = engine.benchmark(original, (done, total) -> {
                setProgressValue(done, total);
                String phase = done < 5 ? "chạy thử " + (done + 1) + "/5"
                        : done < 55 ? "đang đo " + (done - 4) + "/50" : "hoàn tất 50/50";
                showProgress((gpu ? "GPU" : "CPU") + " · " + phase,
                        "Đo " + (gpu ? "GPU" : "CPU") + " · " + phase);
            });
            double sum = 0;
            JSONArray raw = new JSONArray();
            for (double s : samples) {
                sum += s;
                raw.put(s);
            }
            value.put("status", "OK").put("mean_ms", sum / 50).put("samples_ms", raw);
            String logs = gpu ? readGpuLogs(since) : "";
            int ops = ModelGraph.operatorCount(restorationModel);
            value.put("coverage", gpu ? GpuEvidence.classify(logs, ops) : "CPU_ONLY")
                    .put("original_single_graph_ops", ops)
                    .put("gpu_log", logs);
        } catch (CancellationException e) {
            throw e;
        } catch (RuntimeException | Error e) {
            if (e instanceof ThreadDeath)
                throw e;
            checkCancelled();
            value.put("status", "FAILED").put("error", e.toString())
                    .put("coverage", gpu ? "GPU_ATTEMPT_FAILED" : "CPU_FAILED");
        }
        return value;
    }

    private void benchmark() {
        job(true, () -> {
            requireImage();
            show("Đang đo CPU: chạy thử 5 lần, đo 50 lần trên tile 256×256…");
            JSONObject cpu = measure(false);
            checkCancelled();
            show("Đã xong CPU. Đang chạy thử và đo GPU…");
            JSONObject gpu = measure(true);
            checkCancelled();
            JSONObject b = new JSONObject()
                    .put("device", Build.MANUFACTURER + " " + Build.MODEL)
                    .put("android_api", Build.VERSION.SDK_INT)
                    .put("model_sha256", sha(restorationModel))
                    .put("cpu_threads", Restorer.cpuThreads())
                    .put("fp32_io", true)
                    .put("cpu_xnnpack", false)
                    .put("gpu_precision_loss_allowed", false)
                    .put("cpu", cpu)
                    .put("gpu_attempt", gpu)
                    .put("note",
                            "FULL/PARTIAL/CPU_ONLY chỉ xác nhận từ log runtime. UNKNOWN cần profiling; trường hợp không xuất hiện trên thiết bị này là N/A.");
            JSONObject scenarios = new JSONObject()
                    .put("FULL_GPU", "N/A")
                    .put("PARTIAL_GPU", "N/A")
                    .put("GPU_UNAVAILABLE_CPU", "N/A");
            String coverage = gpu.optString("coverage");
            if (coverage.equals("FULL_GPU") || coverage.equals("PARTIAL_GPU"))
                scenarios.put(coverage, gpu);
            if ((gpu.optString("status").equals("FAILED") || coverage.equals("CPU_ONLY"))
                    && cpu.optString("status").equals("OK")) {
                scenarios.put("GPU_UNAVAILABLE_CPU", new JSONObject().put("cpu_baseline", cpu).put("gpu_attempt", gpu));
            }
            b.put("scenarios", scenarios);
            report.put("benchmark", b);
            String cpuTime = cpu.has("mean_ms") ? String.format(Locale.US, "%.2f ms", cpu.getDouble("mean_ms")) : "lỗi";
            String gpuTime = gpu.has("mean_ms") ? String.format(Locale.US, "%.2f ms", gpu.getDouble("mean_ms"))
                    : "không chạy được";
            show("Hoàn tất đo 5+50 · CPU: " + cpuTime + " · GPU: " + gpuTime + ". Xem biểu đồ ở thẻ Đánh giá.");
        });
    }

    private void preview(boolean segment) {
        job(() -> {
            requireImage();
            if (restored == null)
                throw new IllegalStateException("Khôi phục ảnh trước");
            show(segment ? "Đang tìm và tô vùng vật thể trên hai ảnh…" : "Đang tìm và khoanh vật thể trên hai ảnh…");
            try (Yolo yolo = new Yolo(segment ? segmentationModel : detectionModel, segment)) {
                List<Instance> before = yolo.predict(original, "preview");
                List<Instance> after = yolo.predict(restored, "preview");
                Bitmap bv = Yolo.draw(original, before);
                Bitmap av = Yolo.draw(restored, after);
                runOnUiThread(() -> {
                    comparator.setImages(bv, av);
                    selectPage(0, true);
                    workspace.smoothScrollTo(0, 0);
                });
                show("Đã " + (segment ? "tô vùng" : "khoanh") + " vật thể (" + before.size() + " → " + after.size()
                        + " vùng). Kéo thanh trượt trên ảnh để đối chiếu.");
            }
        });
    }

    private void evaluate(boolean segment) {
        job(true, () -> {
            if (dataset == null) {
                throw new IllegalStateException("Hãy chọn bộ ảnh kiểm tra (.zip) có đáp án đúng trước khi chấm điểm.");
            }
            for (int i = 0; i < dataset.images.length(); i++) {
                if (dataset.images.getJSONObject(i).optString("annotation_status").equals("pending")) {
                    throw new IllegalStateException(
                            "Bộ ảnh chưa có đáp án đúng. Bạn có thể dùng chức năng khôi phục hàng loạt.");
                }
            }
            Map<String, List<Instance>> gt = new TreeMap<>(), before = new TreeMap<>(), after = new TreeMap<>();
            Map<String, Integer> counts = new TreeMap<>();
            JSONArray samples = new JSONArray();
            report.remove("detection");
            report.remove("segmentation");
            report.remove("batch_restoration");
            File folder = new File(getFilesDir(), "evaluation-images");
            folder.mkdirs();
            for (File f : Objects.requireNonNull(folder.listFiles()))
                f.delete();
            File yoloFile = segment ? segmentationModel : detectionModel;
            int totalImages = dataset.images.length();
            try (Restorer engine = openRestorer(restorationModel, false);
                    Yolo yolo = new Yolo(yoloFile, segment)) {
                for (int i = 0; i < totalImages; i++) {
                    checkCancelled();
                    final int idx = i;
                    JSONObject row = dataset.images.getJSONObject(i);
                    String name = row.getString("degradation"), id = row.getString("id");
                    JSONObject sample = new JSONObject(row.toString()).put("input_sha256",
                            sha(dataset.file(row.getString("input"))));
                    samples.put(sample);
                    String baseLabel = "Ảnh " + (i + 1) + "/" + totalImages + " · " + taskName(name);
                    setProgressValue(i * 100, totalImages * 100);
                    showProgress(baseLabel, "Đánh giá " + baseLabel + " · " + (segment ? "tô vùng" : "khoanh vật thể"));
                    File sourceFile = dataset.file(row.getString("input"));
                    BitmapFactory.Options bounds = new BitmapFactory.Options();
                    bounds.inJustDecodeBounds = true;
                    BitmapFactory.decodeFile(sourceFile.getPath(), bounds);
                    if ((long) bounds.outWidth * bounds.outHeight > 24000000)
                        throw new IOException("Ảnh >24 megapixel");
                    Bitmap source = BitmapFactory.decodeFile(sourceFile.getPath());
                    if (source == null)
                        throw new IOException("Không đọc được ảnh " + id);
                    if ((long) source.getWidth() * source.getHeight() > 24000000) {
                        source.recycle();
                        throw new IOException("Ảnh >24 megapixel");
                    }
                    try {
                        List<Instance> truth = TestSet.annotations(row, source.getWidth(), source.getHeight(), segment);
                        gt.computeIfAbsent(name, k -> new ArrayList<>()).addAll(truth);
                        List<Instance> bp = yolo.predict(source, id);
                        before.computeIfAbsent(name, k -> new ArrayList<>()).addAll(bp);
                        sample.put("before_predictions", predictions(bp));
                        Bitmap result = engine.restore(source, (done, total) -> {
                            int pct = total > 0 ? (done * 100 / total) : 0;
                            setProgressValue(idx * 100 + pct, totalImages * 100);
                            String tileInfo = "tile " + Math.min(total, done + 1) + "/" + total;
                            showProgress(baseLabel + " · " + tileInfo,
                                    "Đánh giá " + baseLabel + " (" + tileInfo + ")");
                        });
                        try {
                            List<Instance> ap = yolo.predict(result, id);
                            after.computeIfAbsent(name, k -> new ArrayList<>()).addAll(ap);
                            sample.put("after_predictions", predictions(ap));
                            try (OutputStream out = new FileOutputStream(
                                    new File(folder, String.format(Locale.US, "%04d-%s.png", i, name)))) {
                                result.compress(Bitmap.CompressFormat.PNG, 100, out);
                            }
                        } finally {
                            result.recycle();
                        }
                    } finally {
                        source.recycle();
                    }
                    counts.put(name, counts.getOrDefault(name, 0) + 1);
                }
            }
            setProgressValue(totalImages * 100, totalImages * 100);
            JSONObject metrics = new JSONObject();
            for (String name : new TreeSet<>(TestSet.TASKS)) {
                if (!gt.containsKey(name) || gt.get(name).isEmpty()) {
                    metrics.put(name, new JSONObject().put("status", "NO_GROUND_TRUTH_OR_IMAGES"));
                    continue;
                }
                double[] b = MapMetric.compute(gt.get(name), before.get(name), segment);
                double[] a = MapMetric.compute(gt.get(name), after.get(name), segment);
                metrics.put(name, new JSONObject()
                        .put("images", counts.get(name))
                        .put("before_mAP50", b[0]).put("after_mAP50", a[0]).put("delta_mAP50", a[0] - b[0])
                        .put("before_mAP50_95", b[1]).put("after_mAP50_95", a[1]).put("delta_mAP50_95", a[1] - b[1]));
            }
            JSONObject result = new JSONObject()
                    .put("task", segment ? "instance_segmentation" : "object_detection")
                    .put("restoration_model_sha256", sha(restorationModel))
                    .put("yolo_model_sha256", sha(yoloFile))
                    .put("protocol",
                            "101-point AP; IoU .50:.05:.95; conf .001; class-aware NMS .7; max300; GT-present classes; no crowd/ignore; polygon raster masks")
                    .put("restoration_backend", "CPU")
                    .put("samples", samples)
                    .put("results", metrics);
            report.put(segment ? "segmentation" : "detection", result);
            show("Đã chấm điểm xong " + totalImages + " ảnh. Xem bảng điểm chi tiết bên dưới.");
        });
    }

    private String batchSignature(TestSet set, File model) throws Exception {
        StringBuilder sb = new StringBuilder(sha(model)).append('|').append(set.images.length());
        for (int i = 0; i < set.images.length(); i++) {
            JSONObject row = set.images.getJSONObject(i);
            sb.append('|').append(row.getString("id")).append(':').append(row.getString("input"));
        }
        return sb.toString();
    }

    private void writeCheckpoint(File file, String signature, JSONArray samples) {
        try {
            JSONObject cp = new JSONObject().put("signature", signature).put("samples", samples);
            java.nio.file.Files.write(file.toPath(), cp.toString().getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            android.util.Log.w("DSwinIR", "Cannot save batch checkpoint", e);
        }
    }

    private void restoreBatch() {
        job(true, () -> {
            if (dataset == null)
                throw new IllegalStateException("Chọn bộ ảnh kiểm tra (.zip) trước");
            report.remove("detection");
            report.remove("segmentation");
            report.remove("batch_restoration");
            File folder = new File(getFilesDir(), "evaluation-images");
            folder.mkdirs();
            File checkpointFile = new File(folder, ".checkpoint.json");
            String sig = batchSignature(dataset, restorationModel);
            JSONArray samples = new JSONArray();
            int startIndex = 0;
            if (checkpointFile.isFile()) {
                try {
                    JSONObject saved = new JSONObject(new String(
                            java.nio.file.Files.readAllBytes(checkpointFile.toPath()), StandardCharsets.UTF_8));
                    if (sig.equals(saved.optString("signature"))) {
                        JSONArray prev = saved.optJSONArray("samples");
                        if (prev != null && prev.length() > 0 && prev.length() <= dataset.images.length()) {
                            boolean allPresent = true;
                            for (int k = 0; k < prev.length(); k++) {
                                String outName = prev.getJSONObject(k).optString("output");
                                if (outName.isEmpty() || !new File(folder, outName).isFile()) {
                                    allPresent = false;
                                    break;
                                }
                            }
                            if (allPresent) {
                                samples = prev;
                                startIndex = prev.length();
                            }
                        }
                    }
                } catch (Exception ignored) {
                    /* start fresh */ }
            }
            if (startIndex == 0) {
                for (File f : Objects.requireNonNull(folder.listFiles()))
                    f.delete();
            }
            int totalImages = dataset.images.length();
            if (startIndex < totalImages) {
                try (Restorer engine = openRestorer(restorationModel, false)) {
                    for (int i = startIndex; i < totalImages; i++) {
                        checkCancelled();
                        final int idx = i;
                        JSONObject row = dataset.images.getJSONObject(i);
                        String deg = row.getString("degradation");
                        String baseLabel = "Ảnh " + (i + 1) + "/" + totalImages + " · " + taskName(deg);
                        setProgressValue(i * 100, totalImages * 100);
                        showProgress(baseLabel,
                                (startIndex > 0 && i == startIndex ? "Tiếp tục · " : "Khôi phục ") + baseLabel);
                        File input = dataset.file(row.getString("input"));
                        BitmapFactory.Options bounds = new BitmapFactory.Options();
                        bounds.inJustDecodeBounds = true;
                        BitmapFactory.decodeFile(input.getPath(), bounds);
                        if ((long) bounds.outWidth * bounds.outHeight > 24000000)
                            throw new IOException("Ảnh >24 megapixel");
                        Bitmap source = BitmapFactory.decodeFile(input.getPath());
                        if (source == null)
                            throw new IOException("Không đọc được ảnh test");
                        try {
                            long start = SystemClock.elapsedRealtimeNanos();
                            Bitmap result = engine.restore(source, (done, total) -> {
                                int pct = total > 0 ? (done * 100 / total) : 0;
                                setProgressValue(idx * 100 + pct, totalImages * 100);
                                String tileInfo = "tile " + Math.min(total, done + 1) + "/" + total;
                                showProgress(baseLabel + " · " + tileInfo,
                                        "Khôi phục " + baseLabel + " (" + tileInfo + ")");
                            });
                            try {
                                double elapsed = (SystemClock.elapsedRealtimeNanos() - start) / 1e6;
                                String filename = String.format(Locale.US, "%04d-%s.png", i, deg);
                                try (OutputStream out = new FileOutputStream(new File(folder, filename))) {
                                    result.compress(Bitmap.CompressFormat.PNG, 100, out);
                                }
                                samples.put(new JSONObject(row.toString())
                                        .put("input_sha256", sha(input))
                                        .put("output", filename)
                                        .put("width", result.getWidth())
                                        .put("height", result.getHeight())
                                        .put("restore_ms", elapsed));
                                writeCheckpoint(checkpointFile, sig, samples);
                                if (i == totalImages - 1) {
                                    original = source.copy(Bitmap.Config.ARGB_8888, false);
                                    restored = result.copy(Bitmap.Config.ARGB_8888, false);
                                    runOnUiThread(() -> comparator.setImages(original, restored));
                                }
                            } finally {
                                result.recycle();
                            }
                        } finally {
                            source.recycle();
                        }
                    }
                }
            }
            checkpointFile.delete();
            setProgressValue(totalImages * 100, totalImages * 100);
            report.put("batch_restoration", new JSONObject()
                    .put("model_sha256", sha(restorationModel))
                    .put("backend", "CPU")
                    .put("timing", "full image restoration; excludes shared interpreter initialization")
                    .put("samples", samples)
                    .put("mAP", "NOT_EVALUATED_WITHOUT_GROUND_TRUTH"));
            show("Đã khôi phục " + samples.length()
                    + " ảnh giữ kích thước gốc. Mở Khôi phục để xem ảnh cuối hoặc lưu tệp .zip.");
        });
    }

    private static JSONArray predictions(List<Instance> values) throws JSONException {
        JSONArray out = new JSONArray();
        for (Instance p : values) {
            JSONArray box = new JSONArray();
            for (float x : p.box)
                box.put(x);
            JSONObject item = new JSONObject().put("class_id", p.cls).put("score", p.score).put("bbox", box);
            if (p.mask != null) {
                JSONArray runs = new JSONArray();
                int start = p.mask.nextSetBit(0);
                while (start >= 0) {
                    int end = p.mask.nextClearBit(start);
                    runs.put(start);
                    runs.put(end - start);
                    start = p.mask.nextSetBit(end);
                }
                item.put("mask_row_major_start_length", runs);
            }
            out.put(item);
        }
        return out;
    }

    private void export() {
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT)
                .setType("application/zip")
                .addCategory(Intent.CATEGORY_OPENABLE)
                .putExtra(Intent.EXTRA_TITLE, "dswinir-report.zip");
        startActivityForResult(intent, EXPORT);
    }

    @Override
    public void onDestroy() {
        ticker.removeCallbacks(tick);
        if (busy && stoppable)
            cancelJob();
        worker.shutdown();
        super.onDestroy();
    }
}
