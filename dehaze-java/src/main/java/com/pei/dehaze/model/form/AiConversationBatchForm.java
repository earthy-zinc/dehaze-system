package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.List;
import lombok.Data;

/**
 * 批量操作会话请求
 *
 * @author dehaze
 */
@Data
public class AiConversationBatchForm {

    @NotBlank(message = "批量操作类型不能为空")
    @Schema(description = "批量操作类型(archive/restore/delete)")
    private String action;

    @NotEmpty(message = "会话ID列表不能为空")
    @Schema(description = "会话ID列表")
    private List<Long> ids;

    @Schema(description = "批量删除二次确认标记")
    private Boolean confirm;

}
