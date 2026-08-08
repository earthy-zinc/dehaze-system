package com.pei.dehaze.ui.compare;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.Nullable;

/**
 * 放大镜对比视图：手指在图上滑动时，以触摸点为中心放大显示原图和去雾图。
 *
 * 设计：屏幕上半为原图放大区，下半为去雾图放大区，中间画分隔线和十字定位。
 * 放大区域取自源图以触摸点为中心的 SRCDATA_RECT 区域，绘制到 TARGET_RECT。
 */
public class MagnifierView extends View {

    private static final int MAGNIFY_FACTOR = 3;     // 放大倍数
    private static final int LENS_RADIUS_DP = 80;    // 放大镜半径
    private static final int DIVIDER_HEIGHT_DP = 1;   // 分隔线高度

    private Bitmap originalBitmap;
    private Bitmap dehazedBitmap;

    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint dividerPaint = new Paint();
    private final Paint crosshairPaint = new Paint();

    private float touchX = -1;
    private float touchY = -1;
    private int lensRadiusPx;

    private final Rect srcRect = new Rect();
    private final RectF dstRectTop = new RectF();
    private final RectF dstRectBottom = new RectF();

    public MagnifierView(Context context) {
        this(context, null);
    }

    public MagnifierView(Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public MagnifierView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        float density = getResources().getDisplayMetrics().density;
        lensRadiusPx = (int) (LENS_RADIUS_DP * density);

        dividerPaint.setColor(Color.argb(180, 255, 255, 255));
        dividerPaint.setStrokeWidth(DIVIDER_HEIGHT_DP * density);

        crosshairPaint.setColor(Color.argb(200, 255, 255, 255));
        crosshairPaint.setStrokeWidth(2 * density);
        crosshairPaint.setStyle(Paint.Style.STROKE);
    }

    public void setOriginalBitmap(Bitmap bitmap) {
        this.originalBitmap = bitmap;
        invalidate();
    }

    public void setDehazedBitmap(Bitmap bitmap) {
        this.dehazedBitmap = bitmap;
        invalidate();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN || event.getAction() == MotionEvent.ACTION_MOVE) {
            touchX = event.getX();
            touchY = event.getY();
            invalidate();
            return true;
        } else if (event.getAction() == MotionEvent.ACTION_UP) {
            touchX = touchY = -1;
            invalidate();
            return true;
        }
        return super.onTouchEvent(event);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        // 上半区 / 下半区
        float halfH = h / 2f;
        dstRectTop.set(0, 0, w, halfH);
        dstRectBottom.set(0, halfH, w, h);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth();
        int h = getHeight();
        float halfH = h / 2f;

        // 分隔线
        canvas.drawLine(0, halfH, w, halfH, dividerPaint);

        if (touchX < 0 || touchY < 0) {
            // 提示用户触摸
            paint.setColor(Color.WHITE);
            paint.setTextAlign(Paint.Align.CENTER);
            paint.setTextSize(14 * getResources().getDisplayMetrics().density);
            canvas.drawText("手指在图上滑动以放大对比", w / 2f, h / 2f, paint);
            return;
        }

        // 计算上半区中心 / 下半区中心（触摸点落在上半或下半决定哪个为主）
        boolean inTopHalf = touchY < halfH;
        float anchorY = inTopHalf ? touchY : touchY - halfH;

        drawMagnifiedRegion(canvas, originalBitmap, dstRectTop, anchorY);
        drawMagnifiedRegion(canvas, dehazedBitmap, dstRectBottom, anchorY);

        // 十字定位
        float cx = touchX;
        canvas.drawLine(cx - 20, halfH - 20, cx + 20, halfH + 20, crosshairPaint);
        canvas.drawLine(cx - 20, halfH + 20, cx + 20, halfH - 20, crosshairPaint);
    }

    /**
     * 在 dstRect 区域内放大显示 srcBitmap 以 (touchX, anchorY) 为中心的区域。
     */
    private void drawMagnifiedRegion(Canvas canvas, Bitmap bitmap, RectF dstRect, float anchorY) {
        if (bitmap == null) return;

        // 1. 将触摸点映射回源图坐标
        float srcW = bitmap.getWidth();
        float srcH = bitmap.getHeight();
        float dstW = dstRect.width();
        float dstH = dstRect.height();
        if (dstW <= 0 || dstH <= 0) return;

        // 屏幕坐标 → 源图坐标（按 dstRect 比例缩放）
        float srcTouchX = touchX / dstW * srcW;
        float srcTouchY = anchorY / dstH * srcH;

        // 2. 计算源图裁剪区域（以触摸点为中心，半径为源图对应的 lensRadius）
        float srcLensX = lensRadiusPx / dstW * srcW / MAGNIFY_FACTOR;
        float srcLensY = lensRadiusPx / dstH * srcH / MAGNIFY_FACTOR;

        // 钳制到源图边界
        float left = clamp(srcTouchX - srcLensX, 0, srcW - 2 * srcLensX);
        float top = clamp(srcTouchY - srcLensY, 0, srcH - 2 * srcLensY);
        float right = left + 2 * srcLensX;
        float bottom = top + 2 * srcLensY;
        if (right > srcW) {
            right = srcW;
            left = right - 2 * srcLensX;
        }
        if (bottom > srcH) {
            bottom = srcH;
            top = bottom - 2 * srcLensY;
        }

        srcRect.set((int) left, (int) top, (int) right, (int) bottom);

        // 3. 绘制放大区域（铺满 dstRect，因 dstRect 比 srcRect 大，自然放大）
        canvas.drawBitmap(bitmap, srcRect, dstRect, paint);

        // 4. 放大镜边框（圆形描边）
        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.argb(150, 255, 255, 255));
        paint.setStrokeWidth(3);
        float lensCx = touchX;
        float lensCy = dstRect.top + anchorY;
        canvas.drawCircle(lensCx, lensCy, lensRadiusPx, paint);
        paint.setStyle(Paint.Style.FILL);
    }

    private static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(v, max));
    }
}
