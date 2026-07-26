package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.common.enums.AlgorithmStatusEnum;
import com.pei.dehaze.model.form.AlgorithmForm;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.importexport.ImportHandler;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 算法导入处理器
 * <p>仅导入算法元数据，不含权重文件。导入后为草稿状态。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class AlgorithmImportHandler implements ImportHandler {

    private final SysAlgorithmService algorithmService;

    @Override
    public String getModule() {
        return "algorithm";
    }

    @Override
    public List<ImportFieldConfig> getFieldConfigs() {
        return List.of(
                ImportFieldConfig.builder().field("name").label("算法名称").required(true).maxLength(50).build(),
                ImportFieldConfig.builder().field("type").label("算法类型").required(true).build(),
                ImportFieldConfig.builder().field("parentId").label("父算法ID(0为顶级)").required(false).build(),
                ImportFieldConfig.builder().field("path").label("模型文件路径").required(false).build(),
                ImportFieldConfig.builder().field("importPath").label("导入路径").required(false).build(),
                ImportFieldConfig.builder().field("description").label("描述").required(false).build(),
                ImportFieldConfig.builder().field("version").label("版本").required(false).build()
        );
    }

    @Override
    public List<Map<String, Object>> getTemplateSampleData() {
        return List.of(Map.of(
                "name", "示例去雾算法",
                "type", "image_dehaze",
                "parentId", "0",
                "path", "/models/example.pth",
                "importPath", "algorithms.example",
                "description", "示例算法",
                "version", "1.0.0"
        ));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ImportResult importBatch(List<Map<String, Object>> rows, ImportOptions options, ProgressCallback callback) {
        boolean partial = options.isPartialMode();
        List<ImportResult.ImportError> errors = new ArrayList<>();
        int successCount = 0;
        int failureCount = 0;
        int total = rows.size();

        for (int i = 0; i < total; i++) {
            int rowNum = i + 2;
            Map<String, Object> row = rows.get(i);
            callback.updateProgress(i + 1, total, "导入第 " + rowNum + " 行");

            try {
                String name = getAsString(row, "name");
                if (CharSequenceUtil.isBlank(name)) {
                    throw new IllegalArgumentException("算法名称为空");
                }
                String type = getAsString(row, "type");
                if (CharSequenceUtil.isBlank(type)) {
                    throw new IllegalArgumentException("算法类型为空");
                }

                // 检查名称是否已存在
                boolean nameExists = algorithmService.getAllAlgorithms().stream()
                        .anyMatch(a -> name.equals(a.getName()));
                if (nameExists) {
                    throw new IllegalArgumentException("算法名称已存在: " + name);
                }

                AlgorithmForm form = new AlgorithmForm();
                form.setName(name);
                form.setType(type);
                form.setParentId(parseLong(row, "parentId", 0L));
                form.setPath(getAsString(row, "path"));
                form.setImportPath(getAsString(row, "importPath"));
                form.setDescription(getAsString(row, "description"));
                form.setStatus(AlgorithmStatusEnum.DRAFT.getValue());

                algorithmService.addAlgorithm(form);
                successCount++;
            } catch (Exception e) {
                failureCount++;
                errors.add(ImportResult.ImportError.builder()
                        .row(rowNum)
                        .message(e.getMessage())
                        .build());
                if (!partial) {
                    throw new RuntimeException("第 " + rowNum + " 行导入失败: " + e.getMessage()
                            + "（全量模式已回滚所有数据）", e);
                }
            }
        }

        return ImportResult.builder()
                .totalRows(total)
                .successCount(successCount)
                .failureCount(failureCount)
                .skippedCount(0)
                .errors(errors)
                .build();
    }

    private Long parseLong(Map<String, Object> row, String key, Long defaultValue) {
        String v = getAsString(row, key);
        if (CharSequenceUtil.isBlank(v)) {
            return defaultValue;
        }
        try {
            return Long.valueOf(v);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private String getAsString(Map<String, Object> row, String key) {
        Object v = row.get(key);
        return v == null ? null : String.valueOf(v).trim();
    }
}
