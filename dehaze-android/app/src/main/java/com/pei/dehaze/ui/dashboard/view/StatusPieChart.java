package com.pei.dehaze.ui.dashboard.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.task.TaskStatus;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 任务状态分布饼图（自绘 Canvas，轻量无三方依赖）。
 */
public class StatusPieChart extends View {

    private final Paint arcPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint legendPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Map<TaskStatus, Long> data = new LinkedHashMap<>();
    private long total = 0;

    // 状态对应颜色
    private static final int[] COLORS = {
            0xFF2196F3, // PENDING - 蓝
            0xFFFF9800, // PROCESSING - 橙
            0xFF4CAF50, // COMPLETED - 绿
            0xFFF44336, // FAILED - 红
            0xFF9E9E9E, // CANCELLED - 灰
    };

    private final float density;
    private final int textColor;

    public StatusPieChart(Context context) {
        this(context, null);
    }

    public StatusPieChart(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        density = context.getResources().getDisplayMetrics().density;

        textColor = ContextCompat.getColor(context, R.color.text_regular);

        textPaint.setTextSize(12f * density);
        textPaint.setColor(textColor);
        textPaint.setTextAlign(Paint.Align.CENTER);

        legendPaint.setTextSize(11f * density);
        legendPaint.setColor(textColor);
    }

    public void setData(Map<TaskStatus, Long> distribution) {
        data.clear();
        total = 0;
        if (distribution != null) {
            // 按固定顺序排列
            for (TaskStatus status : TaskStatus.values()) {
                Long count = distribution.getOrDefault(status, 0L);
                data.put(status, count);
                total += count;
            }
        }
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int w = getWidth() - getPaddingLeft() - getPaddingRight();
        int h = getHeight() - getPaddingTop() - getPaddingBottom();
        if (w <= 0 || h <= 0) return;

        float cx = getPaddingLeft() + w * 0.38f;
        float cy = getPaddingTop() + h * 0.48f;
        float radius = Math.min(w * 0.35f, h * 0.42f);

        if (total == 0) {
            textPaint.setTextAlign(Paint.Align.CENTER);
            canvas.drawText("暂无状态分布数据", getWidth() / 2f, getHeight() / 2f, textPaint);
            return;
        }

        // 绘制饼图扇形
        RectF oval = new RectF(cx - radius, cy - radius, cx + radius, cy + radius);
        float startAngle = -90f; // 从顶部开始

        int colorIdx = 0;
        for (Map.Entry<TaskStatus, Long> entry : data.entrySet()) {
            long count = entry.getValue();
            if (count <= 0) {
                colorIdx++;
                continue;
            }
            float sweepAngle = 360f * count / total;

            arcPaint.setColor(COLORS[colorIdx % COLORS.length]);
            arcPaint.setStyle(Paint.Style.FILL);
            canvas.drawArc(oval, startAngle, sweepAngle, true, arcPaint);

            // 扇形边框
            arcPaint.setStyle(Paint.Style.STROKE);
            arcPaint.setStrokeWidth(1.5f * density);
            arcPaint.setColor(Color.WHITE);
            canvas.drawArc(oval, startAngle, sweepAngle, true, arcPaint);

            startAngle += sweepAngle;
            colorIdx++;
        }

        // 中心圆孔（甜甜圈效果）
        Paint holePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        holePaint.setColor(Color.WHITE);
        holePaint.setStyle(Paint.Style.FILL);
        canvas.drawCircle(cx, cy, radius * 0.55f, holePaint);

        // 中心文字：总数
        textPaint.setTextSize(13f * density);
        textPaint.setColor(textColor);
        canvas.drawText("共 " + total + " 个", cx, cy - 6f * density, textPaint);
        textPaint.setTextSize(10f * density);
        canvas.drawText("任务", cx, cy + 12f * density, textPaint);

        // 绘制图例（右侧）
        float legendX = getPaddingLeft() + w * 0.72f;
        float legendY = getPaddingTop() + h * 0.12f;
        float legendStep = 22f * density;

        legendPaint.setTextSize(11f * density);
        for (int i = 0; i < TaskStatus.values().length; i++) {
            TaskStatus status = TaskStatus.values()[i];
            Long count = data.getOrDefault(status, 0L);
            float y = legendY + legendStep * i;

            // 色块
            Paint blockPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            blockPaint.setColor(COLORS[i]);
            blockPaint.setStyle(Paint.Style.FILL);
            canvas.drawRect(legendX, y - 6f * density, legendX + 10f * density, y + 4f * density, blockPaint);

            // 文字
            legendPaint.setTextAlign(Paint.Align.LEFT);
            String label = status.getLabel() + " " + count;
            canvas.drawText(label, legendX + 14f * density, y + 4f * density, legendPaint);
        }
    }
}
