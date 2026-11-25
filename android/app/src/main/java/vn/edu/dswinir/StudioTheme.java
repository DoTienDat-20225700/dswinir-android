package vn.edu.dswinir;

import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.StateListAnimator;
import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.graphics.drawable.StateListDrawable;
import android.provider.Settings;
import android.view.View;
import android.view.animation.Interpolator;
import android.view.animation.PathInterpolator;
import android.widget.TextView;

/**
 * Swiss Optical Lab design tokens and adaptive Light/Dark primitives.
 * Uses tinted archival neutrals and high-contrast optical vermilion & teal accents.
 */
public final class StudioTheme {
    public static final Interpolator EASE_OUT_QUINT = new PathInterpolator(0.22f, 1f, 0.36f, 1f);
    public static final Interpolator EASE_OUT_QUART = new PathInterpolator(0.25f, 1f, 0.5f, 1f);

    public static final String PREFS = "studio_prefs";
    public static final String KEY_THEME_MODE = "theme_mode"; // system | light | dark

    public final boolean dark;
    public final int bg;
    public final int surface;
    public final int surfaceSubtle;
    public final int ink;
    public final int inkSecondary;
    public final int muted;
    public final int hairline;
    public final int accent;
    public final int accentInk;
    public final int accentSoft;
    public final int accentSoftInk;
    public final int teal;
    public final int tealSoft;
    public final int tealSoftInk;
    public final int amber;
    public final int amberSoft;
    public final int danger;
    public final int dangerSoft;
    public final int disabledBg;
    public final int disabledInk;
    public final int rippleTint;

    private final float density;
    public final boolean reducedMotion;

    private StudioTheme(Context context) {
        String pref = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(KEY_THEME_MODE, "system");
        int mode = context.getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK;
        if ("dark".equals(pref)) this.dark = true;
        else if ("light".equals(pref)) this.dark = false;
        else this.dark = mode == Configuration.UI_MODE_NIGHT_YES;
        this.density = context.getResources().getDisplayMetrics().density;
        this.reducedMotion = isReducedMotion(context);

        if (dark) {
            bg = 0xFF111316;
            surface = 0xFF191C21;
            surfaceSubtle = 0xFF23272E;
            ink = 0xFFEDEBE6;
            inkSecondary = 0xFFB7BAC0;
            muted = 0xFF8C929C;
            hairline = 0xFF2C313A;
            accent = 0xFFEB6A44;
            accentInk = 0xFF141210;
            accentSoft = 0xFF2E1E1A;
            accentSoftInk = 0xFFF39173;
            teal = 0xFF4CC3BF;
            tealSoft = 0xFF162B2B;
            tealSoftInk = 0xFF7ADAD6;
            amber = 0xFFE8A44C;
            amberSoft = 0xFF2B2216;
            danger = 0xFFEB6B66;
            dangerSoft = 0xFF2E1B1B;
            disabledBg = 0xFF1E2228;
            disabledInk = 0xFF646A75;
            rippleTint = 0x28FFFFFF;
        } else {
            bg = 0xFFF6F4EF;
            surface = 0xFFFCFBF8;
            surfaceSubtle = 0xFFECE9E1;
            ink = 0xFF161A1E;
            inkSecondary = 0xFF454C55;
            muted = 0xFF666D77;
            hairline = 0xFFDCD8CE;
            accent = 0xFFC43812;
            accentInk = 0xFFFCFBF8;
            accentSoft = 0xFFF7E7E2;
            accentSoftInk = 0xFF9B2809;
            teal = 0xFF0B7270;
            tealSoft = 0xFFE3F1F0;
            tealSoftInk = 0xFF085654;
            amber = 0xFFA65708;
            amberSoft = 0xFFFAF0E1;
            danger = 0xFFB92B27;
            dangerSoft = 0xFFFBEAE9;
            disabledBg = 0xFFE5E2DA;
            disabledInk = 0xFF7B8189;
            rippleTint = 0x1A161A1E;
        }
    }

    public static StudioTheme of(Context context) {
        return new StudioTheme(context);
    }

    public static boolean isReducedMotion(Context context) {
        try {
            float scale = Settings.Global.getFloat(context.getContentResolver(), Settings.Global.ANIMATOR_DURATION_SCALE, 1f);
            return scale <= 0.01f;
        } catch (Exception ignored) {
            return false;
        }
    }

    public int dp(int value) {
        return Math.round(value * density);
    }

    public float dpf(float value) {
        return value * density;
    }

    public GradientDrawable rect(int fillColor, int radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(fillColor);
        if (radiusDp > 0) d.setCornerRadius(dpf(radiusDp));
        return d;
    }

    public GradientDrawable outlined(int fillColor, int strokeColor, int strokeDp, int radiusDp) {
        GradientDrawable d = rect(fillColor, radiusDp);
        if (strokeDp > 0 && strokeColor != Color.TRANSPARENT) {
            d.setStroke(Math.max(1, dp(strokeDp)), strokeColor);
        }
        return d;
    }

    public Drawable buttonBg(int normalFill, int strokeColor, int radiusDp) {
        StateListDrawable states = new StateListDrawable();
        states.addState(new int[]{-android.R.attr.state_enabled}, outlined(disabledBg, Color.TRANSPARENT, 0, radiusDp));
        states.addState(new int[]{}, outlined(normalFill, strokeColor, strokeColor == Color.TRANSPARENT ? 0 : 1, radiusDp));
        return new RippleDrawable(ColorStateList.valueOf(rippleTint), states, rect(Color.WHITE, radiusDp));
    }

    public TextView text(Context context, String value, float sizeSp, int color, boolean bold) {
        TextView tv = new TextView(context);
        tv.setText(value);
        tv.setTextSize(sizeSp);
        tv.setTextColor(color);
        if (bold) tv.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        else tv.setTypeface(Typeface.create("sans-serif", Typeface.NORMAL));
        return tv;
    }

    public TextView mono(Context context, String value, float sizeSp, int color) {
        TextView tv = new TextView(context);
        tv.setText(value);
        tv.setTextSize(sizeSp);
        tv.setTextColor(color);
        tv.setFontFeatureSettings("tnum");
        tv.setTypeface(Typeface.MONOSPACE);
        return tv;
    }

    public void attachPressFeedback(View view) {
        if (reducedMotion) return;
        StateListAnimator sla = new StateListAnimator();

        AnimatorSet down = new AnimatorSet();
        down.playTogether(
                ObjectAnimator.ofFloat(view, View.SCALE_X, 0.982f),
                ObjectAnimator.ofFloat(view, View.SCALE_Y, 0.982f)
        );
        down.setDuration(90);
        down.setInterpolator(EASE_OUT_QUART);

        AnimatorSet up = new AnimatorSet();
        up.playTogether(
                ObjectAnimator.ofFloat(view, View.SCALE_X, 1f),
                ObjectAnimator.ofFloat(view, View.SCALE_Y, 1f)
        );
        up.setDuration(180);
        up.setInterpolator(EASE_OUT_QUINT);

        sla.addState(new int[]{android.R.attr.state_pressed, android.R.attr.state_enabled}, down);
        sla.addState(new int[]{}, up);
        view.setStateListAnimator(sla);
    }
}
