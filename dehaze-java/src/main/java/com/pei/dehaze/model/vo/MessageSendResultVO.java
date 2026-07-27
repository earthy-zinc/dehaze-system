package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "消息发送结果")
public class MessageSendResultVO {

    @Schema(description = "生成的消息ID列表")
    private List<Long> messageIds;
}
