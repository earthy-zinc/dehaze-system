package com.pei.dehaze.service.importexport;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/**
 * 导入模板管理器
 * <p>动态生成导入模板（含表头和示例数据），支持 Excel 和 CSV 两种格式。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class TemplateManager {

    private final ImportExportFileGenerator fileGenerator;
    private final ImportHandlerRegistry importHandlerRegistry;

    /**
     * 生成并下载模板
     */
    public void downloadTemplate(String module, String format, HttpServletResponse response) {
        ImportHandler handler = importHandlerRegistry.getHandler(module);
        List<ImportFieldConfig> fields = handler.getDynamicFieldConfigs();
        if (fields == null || fields.isEmpty()) {
            throw new BusinessException(ResultCode.MODULE_IMPORT_NOT_SUPPORTED,
                    "模块 " + module + " 未配置导入字段");
        }

        List<Map<String, Object>> sampleData = handler.getTemplateSampleData();

        String ext = "csv".equalsIgnoreCase(format) ? "csv" : "xlsx";
        String fileName = module + "_import_template." + ext;
        String contentType = "csv".equalsIgnoreCase(format)
                ? "text/csv"
                : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

        response.setContentType(contentType);
        response.setHeader("Content-Disposition",
                "attachment; filename=" + URLEncoder.encode(fileName, StandardCharsets.UTF_8));

        try (OutputStream os = response.getOutputStream()) {
            if ("csv".equalsIgnoreCase(format)) {
                fileGenerator.writeTemplateCsv(os, fields, sampleData);
            } else {
                fileGenerator.writeTemplateExcel(os, fields, sampleData);
            }
        } catch (IOException e) {
            log.error("生成模板失败: module={}, format={}", module, format, e);
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "模板生成失败: " + e.getMessage());
        }
    }
}
