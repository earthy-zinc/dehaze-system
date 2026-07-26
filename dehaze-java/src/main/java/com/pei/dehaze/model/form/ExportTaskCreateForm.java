package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 异步任务创建表单（通用）
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Schema(description = "异步任务创建表单")
@Data
public class ExportTaskCreateForm {

    @Schema(
            description = "任务类型：dataset_export, user_export, role_export, user_import 等",
            example = "user_export"
    )
    @NotBlank(message = "任务类型不能为空")
    private String type;

    @Schema(description = "通用任务参数（JSON 字符串，由导入导出框架解析）",
            example = "{\"module\":\"user\",\"format\":\"excel\",\"query\":{}}")
    private String paramsJson;
}
