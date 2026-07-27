package com.pei.dehaze.utils;

import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.DrawableRes;

import com.pei.dehaze.R;

public final class StatePlaceholder {

    public static final int STATE_EMPTY = 0;
    public static final int STATE_LOADING = 1;
    public static final int STATE_HIDDEN = 2;

    private final View root;
    private final ProgressBar progress;
    private final ImageView icon;
    private final TextView message;

    public StatePlaceholder(View root) {
        this.root = root;
        this.progress = root.findViewById(R.id.state_progress);
        this.icon = root.findViewById(R.id.state_icon);
        this.message = root.findViewById(R.id.state_message);
    }

    public void showEmpty(String text, @DrawableRes int iconRes) {
        root.setVisibility(View.VISIBLE);
        progress.setVisibility(View.GONE);
        icon.setVisibility(View.VISIBLE);
        if (iconRes != 0) {
            icon.setImageResource(iconRes);
        }
        message.setVisibility(View.VISIBLE);
        message.setText(text != null ? text : "暂无数据");
    }

    public void showLoading(String text) {
        root.setVisibility(View.VISIBLE);
        progress.setVisibility(View.VISIBLE);
        icon.setVisibility(View.GONE);
        message.setVisibility(View.VISIBLE);
        message.setText(text != null ? text : "加载中…");
    }

    public void hide() {
        root.setVisibility(View.GONE);
    }
}
