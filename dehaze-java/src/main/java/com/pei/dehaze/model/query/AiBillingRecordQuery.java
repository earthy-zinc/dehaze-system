package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 计费明细查询参数（dateStart/dateEnd 为字符串，按 python 口径解析后抛 A0400） */
@Schema(description = "计费明细查询参数")
@Data
public class AiBillingRecordQuery {

    @Min(1)
    private Long userId;

    @Min(1)
    private Integer pageNum = 1;

    @Min(1)
    @Max(100)
    private Integer pageSize = 20;

    private Long conversationId;

    private String billType;

    private String modelId;

    private String dateStart;

    private String dateEnd;
}
