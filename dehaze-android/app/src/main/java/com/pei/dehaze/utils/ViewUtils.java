package com.pei.dehaze.utils;

import android.view.View;
import android.widget.TextView;

/**
 * View 工具类，消除项目中重复的 setText 等方法
 */
public final class ViewUtils {
    private ViewUtils() {}

    /** 安全设置 TextView 文本，null 显示 fallback */
    public static void setText(View root, int viewId, String text) {
        TextView tv = root.findViewById(viewId);
        if (tv != null) {
            tv.setText(text != null ? text : "—");
        }
    }

    /** 安全设置 TextView 文本，空字符串也显示 fallback */
    public static void setText(View root, int viewId, String text, String fallback) {
        TextView tv = root.findViewById(viewId);
        if (tv != null) {
            tv.setText(text != null && !text.isEmpty() ? text : fallback);
        }
    }
}
