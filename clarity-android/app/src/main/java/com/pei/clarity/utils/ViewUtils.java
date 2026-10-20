package com.pei.clarity.utils;

import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.TextView;

import com.pei.clarity.sdk.model.ai_model.AiModelVO;

import java.util.ArrayList;
import java.util.List;

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

    /**
     * 根据启用模型列表更新模型选择 Spinner 的显示内容（展示 displayName）。
     * 构建 label 列表并设置 ArrayAdapter，models 为 null 时清空。
     */
    public static void updateModelSpinner(Spinner spinner, List<AiModelVO> models) {
        List<String> labels = new ArrayList<>();
        if (models != null) {
            for (AiModelVO model : models) {
                labels.add(model.getDisplayName() != null ? model.getDisplayName() : model.getModelId());
            }
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(spinner.getContext(),
                android.R.layout.simple_spinner_item, labels);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
    }
}
