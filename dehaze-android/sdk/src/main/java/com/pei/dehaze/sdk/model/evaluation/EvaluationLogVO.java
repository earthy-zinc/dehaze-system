package com.pei.dehaze.sdk.model.evaluation;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.util.LinkedHashMap;
import java.util.Map;

import lombok.Data;

/**
 * 评估日志VO（GET /api/v1/evaluation/logs、/metrics）
 */
@Data
public class EvaluationLogVO {
    /** 日志ID */
    private Long id;
    /** 图像模型标识（统一模型规范 slug） */
    private String modelId;
    /** 预测图内容MD5 */
    private String predMd5;
    /** 预测图URL */
    private String predUrl;
    /** 参考图内容MD5 */
    private String gtMd5;
    /** 参考图URL */
    private String gtUrl;
    /** 任务状态（1=处理中，2=已完成，3=失败） */
    private EvalTaskStatus status;
    /** 失败错误信息 */
    private String errorMessage;
    /** 评估耗时（秒） */
    private Integer time;
    /**
     * 评估指标原始值。Python 直接返回对象，Java/Go 返回同一 JSON 的字符串形态，
     * 故保留原始 JSON 并统一由 {@link #getMetrics()} 归一化。
     */
    private JsonElement result;
    /** 创建时间 */
    private String createTime;

    /** 评估指标（PSNR/SSIM 等），无结果时返回 null */
    public Map<String, Double> getMetrics() {
        JsonObject payload = toJsonObject(result);
        if (payload == null) {
            return null;
        }
        Map<String, Double> metrics = new LinkedHashMap<>();
        for (Map.Entry<String, JsonElement> entry : payload.entrySet()) {
            JsonElement value = entry.getValue();
            if (value.isJsonPrimitive() && value.getAsJsonPrimitive().isNumber()) {
                metrics.put(entry.getKey(), value.getAsDouble());
            }
        }
        return metrics;
    }

    private static JsonObject toJsonObject(JsonElement element) {
        if (element == null || element.isJsonNull()) {
            return null;
        }
        if (element.isJsonObject()) {
            return element.getAsJsonObject();
        }
        if (element.isJsonPrimitive()) {
            try {
                JsonElement parsed = JsonParser.parseString(element.getAsString());
                return parsed.isJsonObject() ? parsed.getAsJsonObject() : null;
            } catch (RuntimeException e) {
                return null;
            }
        }
        return null;
    }
}
