package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.form.DictForm;
import com.pei.dehaze.service.SysDictService;
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
 * 字典导入处理器
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class DictImportHandler implements ImportHandler {

    private final SysDictService dictService;

    @Override
    public String getModule() {
        return "dict";
    }

    @Override
    public List<ImportFieldConfig> getFieldConfigs() {
        return List.of(
                ImportFieldConfig.builder().field("typeCode").label("字典类型编码").required(true).maxLength(50).build(),
                ImportFieldConfig.builder().field("name").label("字典名称").required(true).maxLength(50).build(),
                ImportFieldConfig.builder().field("value").label("字典值").required(true).maxLength(50).build(),
                ImportFieldConfig.builder().field("sort").label("排序").required(false).build(),
                ImportFieldConfig.builder().field("statusLabel").label("状态(启用/禁用)").required(false).build(),
                ImportFieldConfig.builder().field("defaulted").label("是否默认(是/否)").required(false).build(),
                ImportFieldConfig.builder().field("remark").label("备注").required(false).build()
        );
    }

    @Override
    public List<Map<String, Object>> getTemplateSampleData() {
        return List.of(Map.of(
                "typeCode", "gender",
                "name", "男",
                "value", "1",
                "sort", "1",
                "statusLabel", "启用",
                "defaulted", "否",
                "remark", ""
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
                String typeCode = getAsString(row, "typeCode");
                if (CharSequenceUtil.isBlank(typeCode)) {
                    throw new IllegalArgumentException("字典类型编码为空");
                }
                String name = getAsString(row, "name");
                if (CharSequenceUtil.isBlank(name)) {
                    throw new IllegalArgumentException("字典名称为空");
                }
                String value = getAsString(row, "value");
                if (CharSequenceUtil.isBlank(value)) {
                    throw new IllegalArgumentException("字典值为空");
                }

                DictForm form = new DictForm();
                form.setTypeCode(typeCode);
                form.setName(name);
                form.setValue(value);
                form.setSort(parseInteger(row, "sort", 0));
                form.setStatus(parseStatus(row, "statusLabel", StatusEnum.ENABLE.getValue()));
                form.setDefaulted(parseBooleanInt(row, "defaulted", 0));
                form.setRemark(getAsString(row, "remark"));

                dictService.saveDict(form);
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

    private Integer parseStatus(Map<String, Object> row, String key, Integer defaultValue) {
        String label = getAsString(row, key);
        if (CharSequenceUtil.isBlank(label)) {
            return defaultValue;
        }
        if ("启用".equals(label)) return StatusEnum.ENABLE.getValue();
        if ("禁用".equals(label)) return StatusEnum.DISABLE.getValue();
        return defaultValue;
    }

    private Integer parseBooleanInt(Map<String, Object> row, String key, Integer defaultValue) {
        String v = getAsString(row, key);
        if (CharSequenceUtil.isBlank(v)) {
            return defaultValue;
        }
        if ("是".equals(v)) return 1;
        if ("否".equals(v)) return 0;
        return defaultValue;
    }

    private Integer parseInteger(Map<String, Object> row, String key, Integer defaultValue) {
        String v = getAsString(row, key);
        if (CharSequenceUtil.isBlank(v)) {
            return defaultValue;
        }
        try {
            return Integer.valueOf(v);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private String getAsString(Map<String, Object> row, String key) {
        Object v = row.get(key);
        return v == null ? null : String.valueOf(v).trim();
    }
}
