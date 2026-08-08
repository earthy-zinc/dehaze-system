package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.List;

/**
 * 前端日志批量上报请求体。
 * <p>
 * 契约见 dehaze-doc/docs/05-改造计划/前端日志监控改造计划.md §3.7.1：
 * 单次请求最多 50 条。
 */
@Data
@Schema(description = "前端日志批量上报表单")
public class ClientLogBatchForm {

    @Schema(description = "日志条目列表（单次最多50条）", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotEmpty(message = "日志列表不能为空")
    @Size(max = 50, message = "单次最多上报50条日志")
    @Valid
    private List<ClientLogEntryForm> logs;
}
