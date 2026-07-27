package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "全部已读结果")
public class ReadAllResultVO {

    @Schema(description = "受影响条数")
    private Integer affectedCount;
}
