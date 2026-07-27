package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "公告发送结果")
public class AnnouncementSendResultVO {

    @Schema(description = "已发送人数")
    private Integer sentCount;
}
