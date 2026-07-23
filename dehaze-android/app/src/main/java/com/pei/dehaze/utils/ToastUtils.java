package com.pei.dehaze.utils;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.widget.Toast;

/**
 * Toast工具类，提供显示Toast的静态方法，确保在主线程中显示
 */
public class ToastUtils {
    private static final Handler handler = new Handler(Looper.getMainLooper());

    private ToastUtils() {
    }

    /**
     * 显示短时间Toast
     */
    public static void showShort(Context context, String message) {
        show(context, message, Toast.LENGTH_SHORT);
    }

    /**
     * 显示长时间Toast
     */
    public static void showLong(Context context, String message) {
        show(context, message, Toast.LENGTH_LONG);
    }

    /**
     * 显示自定义时长Toast，自动处理线程切换
     */
    public static void show(Context context, String message, int duration) {
        if (context == null || message == null) {
            return;
        }
        if (Looper.myLooper() == Looper.getMainLooper()) {
            Toast.makeText(context, message, duration).show();
        } else {
            handler.post(() -> Toast.makeText(context, message, duration).show());
        }
    }
}
