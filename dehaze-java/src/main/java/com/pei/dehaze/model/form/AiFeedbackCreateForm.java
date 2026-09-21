package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.util.List;
import lombok.Data;

/**
 * 提交消息反馈请求
 *
 * @author dehaze
 */
@Data
public class AiFeedbackCreateForm {

    @NotNull(message = "评分不能为空")
    @Schema(description = "评分(1:点赞;-1:点踩)")
    private Integer rating;

    @Schema(description = "预设标签")
    private List<String> tags;

    @Size(max = 2000, message = "反馈内容长度不能超过2000")
    @Schema(description = "反馈内容")
    private String comment;

}
