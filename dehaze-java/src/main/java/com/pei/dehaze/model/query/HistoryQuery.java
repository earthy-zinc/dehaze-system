package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 历史记录查询参数
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "历史记录查询参数")
@Data
public class HistoryQuery {

    @Schema(description = "页码", defaultValue = "1")
    private Integer pageNum = 1;

    @Schema(description = "每页数量", defaultValue = "20")
    private Integer pageSize = 20;

    @Schema(description = "状态筛选（1=成功，2=失败，3=处理中）")
    private Integer status;

    @Schema(description = "来源筛选（upload/camera/sample）")
    private String inputSource;

}
