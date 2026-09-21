package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 更新评测集请求
 *
 * @author dehaze
 */
@Data
public class AiEvalDatasetUpdateForm {

    @Size(max = 128, message = "评测集名称长度不能超过128")
    @Schema(description = "评测集名称")
    private String name;

    @Size(max = 512, message = "评测集描述长度不能超过512")
    @Schema(description = "评测集描述")
    private String description;

}
