package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/** 兼容调用审计查询参数（分页参数为 page/size，与 python `/ai/compat/calls` 一致） */
@Schema(description = "兼容调用审计查询参数")
@Data
public class AiCompatCallQuery {

    @Min(1)
    private Integer page = 1;

    @Min(1)
    @Max(100)
    private Integer size = 20;

    private Long keyId;

    private String model;

    /** 支持 yyyy-MM-dd HH:mm:ss / ISO / yyyy-MM-dd，非法格式按无过滤处理 */
    private String startTime;

    private String endTime;
}
