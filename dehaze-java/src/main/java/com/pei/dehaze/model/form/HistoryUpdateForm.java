package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 历史记录更新表单（如添加收藏）
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "历史记录更新表单")
@Data
public class HistoryUpdateForm {

    @Schema(description = "是否收藏")
    private Boolean isFavorite;
}
