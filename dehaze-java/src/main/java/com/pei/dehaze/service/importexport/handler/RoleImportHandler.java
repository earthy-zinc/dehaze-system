package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.service.SysRoleService;
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
 * 角色导入处理器
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RoleImportHandler implements ImportHandler {

    private final SysRoleService roleService;

    @Override
    public String getModule() {
        return "role";
    }

    @Override
    public List<ImportFieldConfig> getFieldConfigs() {
        return List.of(
                ImportFieldConfig.builder().field("name").label("角色名称").required(true).maxLength(64).build(),
                ImportFieldConfig.builder().field("code").label("角色编码").required(true).maxLength(32).build(),
                ImportFieldConfig.builder().field("sort").label("排序").required(false).build(),
                ImportFieldConfig.builder().field("statusLabel").label("状态(启用/禁用)").required(false).build()
        );
    }

    @Override
    public List<Map<String, Object>> getTemplateSampleData() {
        return List.of(Map.of(
                "name", "普通用户",
                "code", "user",
                "sort", "1",
                "statusLabel", "启用"
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
                    throw new IllegalArgumentException("角色名称为空");
                }
                String code = getAsString(row, "code");
                if (CharSequenceUtil.isBlank(code)) {
                    throw new IllegalArgumentException("角色编码为空");
                }

                long exists = roleService.count(new LambdaQueryWrapper<SysRole>()
                        .eq(SysRole::getCode, code));
                if (exists > 0) {
                    throw new IllegalArgumentException("角色编码已存在: " + code);
                }

                SysRole entity = new SysRole();
                entity.setName(name);
                entity.setCode(code);
                entity.setSort(parseInteger(row, "sort", 0));
                entity.setStatus(parseStatus(row, "statusLabel", StatusEnum.ENABLE.getValue()));
                entity.setDataScope(5); // 默认全部数据权限

                boolean saved = roleService.save(entity);
                if (!saved) {
                    throw new IllegalStateException("保存失败");
                }
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
