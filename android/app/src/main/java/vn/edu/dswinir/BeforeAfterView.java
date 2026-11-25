package vn.edu.dswinir;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.HapticFeedbackConstants;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;

/**
 * Interactive Swiss Optical Lab image comparator supporting Split-wipe, Before, After,
 * pinch/double-tap zoom, and pan inspection.
 */
public final class BeforeAfterView extends View {
    public enum Mode { SPLIT, BEFORE, AFTER }

    private final StudioTheme theme;
    private Bitmap beforeBitmap, afterBitmap;
    private Mode mode = Mode.SPLIT;
    private float splitFraction = 0.5f;
    private float zoom = 1f;
    private float panX = 0f, panY = 0f;
    private boolean draggingDivider;
    private boolean crossedMidpoint;
    private float lastTouchX, lastTouchY;

    private Runnable onEmptyClick;
    private Runnable onZoomChanged;

    private final Paint bitmapPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint reticlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Matrix drawMatrix = new Matrix();
    private final RectF viewRect = new RectF();
    private final RectF fittedRect = new RectF();
    private final Path clipPath = new Path();

    private final ScaleGestureDetector scaleDetector;
    private final GestureDetector gestureDetector;

    public BeforeAfterView(Context context) {
        this(context, null, StudioTheme.of(context));
    }

    public BeforeAfterView(Context context, AttributeSet attrs) {
        this(context, attrs, StudioTheme.of(context));
    }

    public BeforeAfterView(Context context, StudioTheme theme) {
        this(context, null, theme);
    }

    private BeforeAfterView(Context context, AttributeSet attrs, StudioTheme theme) {
        super(context, attrs);
        this.theme = theme;
        bitmapPaint.setFilterBitmap(true);

        textPaint.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        textPaint.setTextSize(theme.dpf(10.5f));

        reticlePaint.setStyle(Paint.Style.STROKE);
        reticlePaint.setStrokeWidth(Math.max(1f, theme.dpf(1f)));

        scaleDetector = new ScaleGestureDetector(context, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override
            public boolean onScale(ScaleGestureDetector detector) {
                if (beforeBitmap == null) return false;
                float prevZoom = zoom;
                zoom = Math.max(1f, Math.min(6f, zoom * detector.getScaleFactor()));
                if (zoom <= 1.01f) {
                    zoom = 1f;
                    panX = 0f;
                    panY = 0f;
                } else {
                    float factor = zoom / prevZoom;
                    float fx = detector.getFocusX() - getWidth() * 0.5f;
                    float fy = detector.getFocusY() - getHeight() * 0.5f;
                    panX = (panX - fx) * factor + fx;
                    panY = (panY - fy) * factor + fy;
                    clampPan();
                }
                if (onZoomChanged != null) onZoomChanged.run();
                invalidate();
                return true;
            }
        });

        gestureDetector = new GestureDetector(context, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onDoubleTap(MotionEvent e) {
                if (beforeBitmap == null) return false;
                if (zoom > 1.1f) {
                    resetZoom();
                } else {
                    zoom = 2.5f;
                    panX = (getWidth() * 0.5f - e.getX()) * 1.5f;
                    panY = (getHeight() * 0.5f - e.getY()) * 1.5f;
                    clampPan();
                    if (onZoomChanged != null) onZoomChanged.run();
                    invalidate();
                }
                performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP);
                return true;
            }

            @Override
            public boolean onSingleTapConfirmed(MotionEvent e) {
                if (beforeBitmap == null) {
                    performClick();
                    return true;
                }
                return false;
            }
        });
    }

    public void setOnEmptyClickListener(Runnable listener) {
        this.onEmptyClick = listener;
    }

    public void setOnZoomChangedListener(Runnable listener) {
        this.onZoomChanged = listener;
    }

    public void setImages(Bitmap before, Bitmap after) {
        this.beforeBitmap = before;
        this.afterBitmap = after;
        if (after == null) {
            mode = Mode.BEFORE;
        } else if (mode == Mode.BEFORE) {
            mode = Mode.SPLIT;
        }
        resetZoom();
        updateContentDescription();
        requestLayout();
        invalidate();
    }

    public void setMode(Mode next) {
        if (next == Mode.SPLIT && afterBitmap == null) return;
        if (next == Mode.AFTER && afterBitmap == null) return;
        this.mode = next;
        updateContentDescription();
        invalidate();
    }

    public Mode getMode() {
        return mode;
    }

    public float getZoom() {
        return zoom;
    }

    public void resetZoom() {
        zoom = 1f;
        panX = 0f;
        panY = 0f;
        if (onZoomChanged != null) onZoomChanged.run();
        invalidate();
    }

    private void updateContentDescription() {
        if (beforeBitmap == null) {
            setContentDescription("Khung xem ảnh trước và sau khôi phục: chưa chọn ảnh");
        } else if (afterBitmap == null) {
            setContentDescription("Ảnh gốc " + beforeBitmap.getWidth() + "×" + beforeBitmap.getHeight());
        } else {
            setContentDescription("So sánh ảnh trước và sau khôi phục " + afterBitmap.getWidth() + "×" + afterBitmap.getHeight());
        }
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        int width = MeasureSpec.getSize(widthMeasureSpec);
        int targetHeight;
        if (beforeBitmap != null && beforeBitmap.getWidth() > 0 && beforeBitmap.getHeight() > 0 && width > 0) {
            float ratio = (float) beforeBitmap.getHeight() / beforeBitmap.getWidth();
            int natural = Math.round(width * ratio);
            targetHeight = Math.max(theme.dp(220), Math.min(theme.dp(420), natural));
        } else {
            targetHeight = theme.dp(230);
        }
        setMeasuredDimension(width, targetHeight);
    }

    private void computeFittedRect(Bitmap bmp) {
        float w = getWidth();
        float h = getHeight();
        float bw = bmp.getWidth();
        float bh = bmp.getHeight();
        float scale = Math.min(w / bw, h / bh);
        float dw = bw * scale;
        float dh = bh * scale;
        float left = (w - dw) * 0.5f;
        float top = (h - dh) * 0.5f;
        fittedRect.set(left, top, left + dw, top + dh);
    }

    private void clampPan() {
        float maxX = Math.max(0f, getWidth() * (zoom - 1f) * 0.5f);
        float maxY = Math.max(0f, getHeight() * (zoom - 1f) * 0.5f);
        panX = Math.max(-maxX, Math.min(maxX, panX));
        panY = Math.max(-maxY, Math.min(maxY, panY));
    }

    @Override
    public boolean performClick() {
        super.performClick();
        if (beforeBitmap == null && onEmptyClick != null) {
            onEmptyClick.run();
            return true;
        }
        return false;
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (beforeBitmap == null) {
            gestureDetector.onTouchEvent(event);
            return true;
        }
        scaleDetector.onTouchEvent(event);
        gestureDetector.onTouchEvent(event);

        int action = event.getActionMasked();
        float x = event.getX();
        float y = event.getY();

        if (event.getPointerCount() > 1) {
            draggingDivider = false;
            getParent().requestDisallowInterceptTouchEvent(true);
            return true;
        }

        switch (action) {
            case MotionEvent.ACTION_DOWN:
                lastTouchX = x;
                lastTouchY = y;
                if (afterBitmap != null && mode == Mode.SPLIT) {
                    float dividerX = getWidth() * splitFraction;
                    if (Math.abs(x - dividerX) <= theme.dpf(32)) {
                        draggingDivider = true;
                        crossedMidpoint = Math.abs(splitFraction - 0.5f) < 0.02f;
                        getParent().requestDisallowInterceptTouchEvent(true);
                        return true;
                    }
                }
                if (zoom > 1.02f) {
                    getParent().requestDisallowInterceptTouchEvent(true);
                }
                return true;

            case MotionEvent.ACTION_MOVE:
                if (draggingDivider) {
                    float next = Math.max(0.03f, Math.min(0.97f, x / Math.max(1f, getWidth())));
                    boolean nearMid = Math.abs(next - 0.5f) < 0.015f;
                    if (nearMid && !crossedMidpoint) {
                        next = 0.5f;
                        crossedMidpoint = true;
                        performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK);
                    } else if (!nearMid) {
                        crossedMidpoint = false;
                    }
                    splitFraction = next;
                    invalidate();
                    return true;
                }
                if (zoom > 1.02f && !scaleDetector.isInProgress()) {
                    float dx = x - lastTouchX;
                    float dy = y - lastTouchY;
                    panX += dx;
                    panY += dy;
                    clampPan();
                    lastTouchX = x;
                    lastTouchY = y;
                    invalidate();
                    return true;
                }
                if (afterBitmap != null && mode == Mode.SPLIT) {
                    float dx = Math.abs(x - lastTouchX);
                    float dy = Math.abs(y - lastTouchY);
                    if (dx > theme.dpf(10) && dx > dy * 1.4f) {
                        draggingDivider = true;
                        getParent().requestDisallowInterceptTouchEvent(true);
                    }
                }
                lastTouchX = x;
                lastTouchY = y;
                return true;

            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                draggingDivider = false;
                getParent().requestDisallowInterceptTouchEvent(false);
                return true;
        }
        return super.onTouchEvent(event);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float w = getWidth();
        float h = getHeight();
        float r = theme.dpf(12);
        viewRect.set(0, 0, w, h);

        clipPath.reset();
        clipPath.addRoundRect(viewRect, r, r, Path.Direction.CW);
        canvas.save();
        canvas.clipPath(clipPath);

        // Recessed optical tray background
        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setColor(theme.surfaceSubtle);
        canvas.drawRect(viewRect, fillPaint);

        if (beforeBitmap == null) {
            drawEmptyViewfinder(canvas, w, h);
            canvas.restore();
            drawFrameBorder(canvas, r);
            return;
        }

        computeFittedRect(beforeBitmap);
        drawMatrix.reset();
        float baseScale = fittedRect.width() / beforeBitmap.getWidth();
        drawMatrix.postScale(baseScale, baseScale);
        drawMatrix.postTranslate(fittedRect.left, fittedRect.top);
        if (zoom > 1.001f) {
            drawMatrix.postScale(zoom, zoom, w * 0.5f, h * 0.5f);
            drawMatrix.postTranslate(panX, panY);
        }

        if (afterBitmap == null || mode == Mode.BEFORE) {
            canvas.drawBitmap(beforeBitmap, drawMatrix, bitmapPaint);
            drawBadge(canvas, theme.dpf(12), theme.dpf(12), "TRƯỚC · GỐC", false);
        } else if (mode == Mode.AFTER) {
            canvas.drawBitmap(afterBitmap, drawMatrix, bitmapPaint);
            drawBadge(canvas, theme.dpf(12), theme.dpf(12), "SAU · KHÔI PHỤC", true);
        } else {
            float splitX = w * splitFraction;
            // Left: Before
            canvas.save();
            canvas.clipRect(0, 0, splitX, h);
            canvas.drawBitmap(beforeBitmap, drawMatrix, bitmapPaint);
            canvas.restore();

            // Right: After
            canvas.save();
            canvas.clipRect(splitX, 0, w, h);
            canvas.drawBitmap(afterBitmap, drawMatrix, bitmapPaint);
            canvas.restore();

            // Corner badges (hide if divider crowds them)
            if (splitX > theme.dpf(92)) {
                drawBadge(canvas, theme.dpf(12), theme.dpf(12), "TRƯỚC", false);
            }
            if (w - splitX > theme.dpf(92)) {
                drawRightBadge(canvas, w - theme.dpf(12), theme.dpf(12), "SAU", true);
            }

            // Optical divider line & handle
            linePaint.setStyle(Paint.Style.STROKE);
            linePaint.setStrokeWidth(theme.dpf(2.5f));
            linePaint.setColor(0x55000000);
            canvas.drawLine(splitX, 0, splitX, h, linePaint);

            linePaint.setStrokeWidth(theme.dpf(1.5f));
            linePaint.setColor(theme.accent);
            canvas.drawLine(splitX, 0, splitX, h, linePaint);

            float cy = h * 0.5f;
            float knobR = theme.dpf(15);
            fillPaint.setColor(theme.surface);
            canvas.drawCircle(splitX, cy, knobR, fillPaint);
            linePaint.setStrokeWidth(theme.dpf(1.5f));
            linePaint.setColor(theme.accent);
            canvas.drawCircle(splitX, cy, knobR, linePaint);

            // Precision chevrons inside handle
            linePaint.setStrokeWidth(theme.dpf(1.6f));
            linePaint.setStrokeCap(Paint.Cap.ROUND);
            linePaint.setColor(theme.ink);
            float aw = theme.dpf(3.2f);
            float ah = theme.dpf(3.8f);
            float gap = theme.dpf(5.8f);
            canvas.drawLine(splitX - gap, cy, splitX - gap + aw, cy - ah, linePaint);
            canvas.drawLine(splitX - gap, cy, splitX - gap + aw, cy + ah, linePaint);
            canvas.drawLine(splitX + gap, cy, splitX + gap - aw, cy - ah, linePaint);
            canvas.drawLine(splitX + gap, cy, splitX + gap - aw, cy + ah, linePaint);
        }

        if (zoom > 1.05f) {
            drawRightBadge(canvas, w - theme.dpf(12), h - theme.dpf(34),
                    String.format(java.util.Locale.US, "%.1f×", zoom), false);
        }

        canvas.restore();
        drawFrameBorder(canvas, r);
    }

    private void drawFrameBorder(Canvas canvas, float r) {
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(Math.max(1f, theme.dpf(1f)));
        linePaint.setColor(theme.hairline);
        canvas.drawRoundRect(viewRect, r, r, linePaint);
    }

    private void drawEmptyViewfinder(Canvas canvas, float w, float h) {
        reticlePaint.setColor(theme.hairline);
        float pad = theme.dpf(22);
        float arm = theme.dpf(16);

        // Four corner L-brackets like an optical rangefinder
        canvas.drawLine(pad, pad, pad + arm, pad, reticlePaint);
        canvas.drawLine(pad, pad, pad, pad + arm, reticlePaint);

        canvas.drawLine(w - pad - arm, pad, w - pad, pad, reticlePaint);
        canvas.drawLine(w - pad, pad, w - pad, pad + arm, reticlePaint);

        canvas.drawLine(pad, h - pad, pad + arm, h - pad, reticlePaint);
        canvas.drawLine(pad, h - pad - arm, pad, h - pad, reticlePaint);

        canvas.drawLine(w - pad - arm, h - pad, w - pad, h - pad, reticlePaint);
        canvas.drawLine(w - pad, h - pad - arm, w - pad, h - pad, reticlePaint);

        // Subtle center reticle
        float cx = w * 0.5f;
        float cy = h * 0.44f;
        float cr = theme.dpf(18);
        reticlePaint.setColor(theme.muted);
        reticlePaint.setAlpha(95);
        canvas.drawCircle(cx, cy, cr, reticlePaint);
        canvas.drawLine(cx - cr - theme.dpf(6), cy, cx - theme.dpf(6), cy, reticlePaint);
        canvas.drawLine(cx + theme.dpf(6), cy, cx + cr + theme.dpf(6), cy, reticlePaint);
        canvas.drawLine(cx, cy - cr - theme.dpf(6), cx, cy - theme.dpf(6), reticlePaint);
        canvas.drawLine(cx, cy + theme.dpf(6), cx, cy + cr + theme.dpf(6), reticlePaint);
        reticlePaint.setAlpha(255);

        Paint titleP = new Paint(Paint.ANTI_ALIAS_FLAG);
        titleP.setColor(theme.ink);
        titleP.setTextAlign(Paint.Align.CENTER);
        titleP.setTextSize(theme.dpf(14f));
        titleP.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        canvas.drawText("Khung soi ảnh quang học", cx, cy + theme.dpf(42), titleP);

        Paint subP = new Paint(Paint.ANTI_ALIAS_FLAG);
        subP.setColor(theme.muted);
        subP.setTextAlign(Paint.Align.CENTER);
        subP.setTextSize(theme.dpf(12f));
        canvas.drawText("Chạm vào đây hoặc nút Chọn ảnh để mở ảnh", cx, cy + theme.dpf(61), subP);
    }

    private void drawBadge(Canvas canvas, float x, float y, String label, boolean accentTeal) {
        float padH = theme.dpf(8);
        float h = theme.dpf(22);
        float tw = textPaint.measureText(label);
        RectF pill = new RectF(x, y, x + tw + padH * 2, y + h);
        fillPaint.setColor(accentTeal ? (theme.dark ? 0xD8162B2B : 0xD80B7270) : 0xCC161A1E);
        canvas.drawRoundRect(pill, theme.dpf(6), theme.dpf(6), fillPaint);
        textPaint.setColor(accentTeal && theme.dark ? theme.tealSoftInk : 0xFFFCFBF8);
        Paint.FontMetrics fm = textPaint.getFontMetrics();
        float ty = pill.centerY() - (fm.ascent + fm.descent) * 0.5f;
        canvas.drawText(label, pill.left + padH, ty, textPaint);
    }

    private void drawRightBadge(Canvas canvas, float rightX, float y, String label, boolean accentTeal) {
        float padH = theme.dpf(8);
        float tw = textPaint.measureText(label);
        drawBadge(canvas, rightX - (tw + padH * 2), y, label, accentTeal);
    }
}
