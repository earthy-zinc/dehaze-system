package com.pei.dehaze.utils;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.widget.Toast;

/**
 * Toast工具类，提供显示Toast的静态方法，确保在主线程中显示
 */
public class ToastUtils {
    private static Handler handler = new Handler(Looper.getMainLooper());

    /**
     * 显示短时间Toast
     */
    public static void showShort(final Context context, final String message) {
        if (context == null || message == null) {
            return;
        }
        
        if (Looper.myLooper() == Looper.getMainLooper()) {
            // 在主线程中直接显示
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show();
        } else {
            // 在子线程中通过Handler切换到主线程显示
            handler.post(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(context, message, Toast.LENGTH_SHORT).show();
                }
            });
        }
    }

    /**
     * 显示长时间Toast
     */
    public static void showLong(final Context context, final String message) {
        if (context == null || message == null) {
            return;
        }
        
        if (Looper.myLooper() == Looper.getMainLooper()) {
            // 在主线程中直接显示
            Toast.makeText(context, message, Toast.LENGTH_LONG).show();
        } else {
            // 在子线程中通过Handler切换到主线程显示
            handler.post(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(context, message, Toast.LENGTH_LONG).show();
                }
            });
        }
    }

    /**
     * 显示自定义时长Toast
     */
    public static void show(final Context context, final String message, final int duration) {
        if (context == null || message == null) {
            return;
        }
        
        if (Looper.myLooper() == Looper.getMainLooper()) {
            // 在主线程中直接显示
            Toast.makeText(context, message, duration).show();
        } else {
            // 在子线程中通过Handler切换到主线程显示
            handler.post(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(context, message, duration).show();
                }
            });
        }
    }
}