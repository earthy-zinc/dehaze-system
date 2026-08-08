package com.pei.dehaze.ui.dashboard.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.pei.dehaze.R;
import com.pei.dehaze.repository.DashboardRepository;

import java.util.ArrayList;
import java.util.List;

/**
 * 近 7 天任务趋势折线图（自绘 Canvas，轻量无三方依赖）。
 */
public class TrendLineChart extends View {

    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint dotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint axisPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final List<DashboardRepository.TrendItem> data = new ArrayList<>();
    private long maxCount = 1;

    private final int lineColor;
    private final int fillColor;
    private final int dotColor;
    private final int gridColor;
    private final int textColor;

    private final float density;

    public TrendLineChart(Context context) {
        this(context, null);
    }

    public TrendLineChart(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        density = context.getResources().getDisplayMetrics().density;

        lineColor = ContextCompat.getColor(context, R.color.brand_primary);
        fillColor = Color.argb(40, Color.red(lineColor), Color.green(lineColor), Color.blue(lineColor));
        dotColor = lineColor;
        gridColor = Color.argb(60, 0, 0, 0);
        textColor = ContextCompat.getColor(context, R.color.text_regular);

        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(3f * density);
        linePaint.setColor(lineColor);
        linePaint.setStrokeCap(Paint.Cap.ROUND);
        linePaint.setStrokeJoin(Paint.Join.ROUND);

        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setColor(fillColor);

        dotPaint.setStyle(Paint.Style.FILL);
        dotPaint.setColor(dotColor);

        gridPaint.setStyle(Paint.Style.STROKE);
        gridPaint.setStrokeWidth(1f * density);
        gridPaint.setColor(gridColor);
        gridPaint.setPathEffect(new DashPathEffect(new float[]{6f * density, 4f * density}, 0));

        textPaint.setTextSize(10f * density);
        textPaint.setColor(textColor);
        textPaint.setTextAlign(Paint.Align.CENTER);

        axisPaint.setStyle(Paint.Style.STROKE);
        axisPaint.setStrokeWidth(1.5f * density);
        axisPaint.setColor(gridColor);
    }

    public void setData(List<DashboardRepository.TrendItem> items) {
        data.clear();
        if (items != null) {
            data.addAll(items);
        }
        maxCount = 1;
        for (DashboardRepository.TrendItem item : data) {
            if (item.getCount() > maxCount) {
                maxCount = item.getCount();
            }
        }
        if (maxCount == 0) maxCount = 1;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int w = getWidth() - getPaddingLeft() - getPaddingRight();
        int h = getHeight() - getPaddingTop() - getPaddingBottom();
        if (w <= 0 || h <= 0) return;

        float left = getPaddingLeft() + 24f * density;   // 左侧留白给 Y 轴标签
        float right = getPaddingLeft() + w - 8f * density;
        float top = getPaddingTop() + 8f * density;
        float bottom = getPaddingTop() + h - 24f * density; // 底部留白给 X 轴标签

        // 绘制网格线和 Y 轴标签
        int gridLines = 4;
        for (int i = 0; i <= gridLines; i++) {
            float y = top + (bottom - top) * i / gridLines;
            canvas.drawLine(left, y, right, y, gridPaint);
            long labelVal = maxCount * (gridLines - i) / gridLines;
            textPaint.setTextAlign(Paint.Align.RIGHT);
            canvas.drawText(String.valueOf(labelVal), left - 4f * density, y + 4f * density, textPaint);
        }

        // 绘制 X 轴
        canvas.drawLine(left, bottom, right, bottom, axisPaint);

        if (data.isEmpty()) {
            textPaint.setTextAlign(Paint.Align.CENTER);
            canvas.drawText("暂无趋势数据", getWidth() / 2f, getHeight() / 2f, textPaint);
            return;
        }

        int n = data.size();
        float stepX = (right - left) / Math.max(n - 1, 1);

        // 计算数据点坐标
        float[] ptsX = new float[n];
        float[] ptsY = new float[n];
        for (int i = 0; i < n; i++) {
            ptsX[i] = left + stepX * i;
            float ratio = maxCount > 0 ? (float) data.get(i).getCount() / maxCount : 0;
            ptsY[i] = bottom - (bottom - top) * ratio;
        }

        // 绘制填充区域
        Path fillPath = new Path();
        fillPath.moveTo(ptsX[0], bottom);
        for (int i = 0; i < n; i++) {
            fillPath.lineTo(ptsX[i], ptsY[i]);
        }
        fillPath.lineTo(ptsX[n - 1], bottom);
        fillPath.close();
        canvas.drawPath(fillPath, fillPaint);

        // 绘制折线
        Path linePath = new Path();
        linePath.moveTo(ptsX[0], ptsY[0]);
        for (int i = 1; i < n; i++) {
            linePath.lineTo(ptsX[i], ptsY[i]);
        }
        canvas.drawPath(linePath, linePaint);

        // 绘制数据点和 X 轴日期标签
        float dotRadius = 4f * density;
        textPaint.setTextAlign(Paint.Align.CENTER);
        for (int i = 0; i < n; i++) {
            canvas.drawCircle(ptsX[i], ptsY[i], dotRadius, dotPaint);

            // X 轴标签：只显示 MM-dd
            String date = data.get(i).getDate();
            String label = date.length() >= 10 ? date.substring(5, 10) : date;
            canvas.drawText(label, ptsX[i], bottom + 16f * density, textPaint);
        }
    }
}
