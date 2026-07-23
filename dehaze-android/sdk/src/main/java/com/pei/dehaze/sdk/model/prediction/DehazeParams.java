package com.pei.dehaze.sdk.model.prediction;

import lombok.Data;

/**
 * 去雾处理通用参数。
 * 范围约束对齐产品文档《去雾处理需求规格》：
 * <ul>
 *     <li>去雾强度 strength：0-100，默认 50</li>
 *     <li>色彩饱和度 saturation：0-200，默认 100（100 为原始值）</li>
 *     <li>对比度 contrast：0-200，默认 100（100 为原始值）</li>
 *     <li>锐化程度 sharpen：0-100，默认 30</li>
 * </ul>
 */
@Data
public class DehazeParams {
    /** 去雾强度 0-100 */
    private int strength;
    /** 色彩饱和度 0-200 */
    private int saturation;
    /** 对比度 0-200 */
    private int contrast;
    /** 锐化程度 0-100 */
    private int sharpen;

    public DehazeParams() {
        this(50, 100, 100, 30);
    }

    public DehazeParams(int strength, int saturation, int contrast, int sharpen) {
        this.strength = strength;
        this.saturation = saturation;
        this.contrast = contrast;
        this.sharpen = sharpen;
    }

    /**
     * 校验参数范围，返回首个不合法字段的错误提示，全部合法返回 null。
     */
    public String validate() {
        if (strength < 0 || strength > 100) {
            return "去雾强度须在 0-100 之间";
        }
        if (saturation < 0 || saturation > 200) {
            return "色彩饱和度须在 0-200 之间";
        }
        if (contrast < 0 || contrast > 200) {
            return "对比度须在 0-200 之间";
        }
        if (sharpen < 0 || sharpen > 100) {
            return "锐化程度须在 0-100 之间";
        }
        return null;
    }
}
