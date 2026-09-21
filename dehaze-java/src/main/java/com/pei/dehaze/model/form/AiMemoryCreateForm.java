package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import java.util.Map;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.Max;
import lombok.Data;

/**
 * 创建记忆请求
 *
 * @author dehaze
 */
@Data
public class AiMemoryCreateForm {

    @NotBlank(message = "记忆类型不能为空")
    @Schema(description = "记忆类型(episodic/semantic/procedural)")
    private String memoryType;

    @NotBlank(message = "记忆内容不能为空")
    @Size(max = 2000, message = "记忆内容长度不能超过2000")
    @Schema(description = "记忆内容")
    private String content;

    @Schema(description = "结构化属性")
    private Map<String, Object> metadata;

    @Min(value = 0, message = "重要性评分不能小于0")
    @Max(value = 100, message = "重要性评分不能大于100")
    @Schema(description = "重要性评分(0-100)")
    private Integer importance;

    @Schema(description = "来源")
    private String source;

}
