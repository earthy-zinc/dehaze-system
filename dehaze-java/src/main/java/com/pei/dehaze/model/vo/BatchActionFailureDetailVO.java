package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(name = "BatchActionFailureDetailVO", description = "批量操作失败详情")
public class BatchActionFailureDetailVO {

    @Schema(description = "失败记录的唯一标识，例如文件ID或分组名称")
    private String identifier;

    @Schema(description = "失败原因")
    private String reason;
}
