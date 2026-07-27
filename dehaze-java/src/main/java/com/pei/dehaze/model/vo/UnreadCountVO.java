package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "未读消息数")
public class UnreadCountVO {

    @Schema(description = "未读数量")
    private Long count;
}
