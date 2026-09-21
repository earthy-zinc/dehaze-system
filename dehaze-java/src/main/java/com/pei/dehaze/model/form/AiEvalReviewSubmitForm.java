package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 复核结果回填请求
 *
 * @author dehaze
 */
@Data
public class AiEvalReviewSubmitForm {

    @NotNull(message = "人工判定不能为空")
    @Schema(description = "人工判定(true:与判分一致;false:不一致)")
    private Boolean agree;

    @Size(max = 500, message = "复核备注长度不能超过500")
    @Schema(description = "复核备注")
    private String remark;

}
