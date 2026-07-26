package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.MenuTypeEnum;
import com.pei.dehaze.model.form.MenuForm;
import com.pei.dehaze.service.SysMenuService;
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
 * 菜单导入处理器
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MenuImportHandler implements ImportHandler {

    private final SysMenuService menuService;

    @Override
    public String getModule() {
        return "menu";
    }

    @Override
    public List<ImportFieldConfig> getFieldConfigs() {
        return List.of(
                ImportFieldConfig.builder().field("name").label("菜单名称").required(true).maxLength(64).build(),
                ImportFieldConfig.builder().field("parentId").label("父菜单ID(0为顶级)").required(false).build(),
                ImportFieldConfig.builder().field("typeLabel").label("类型(菜单/目录/外链/按钮)").required(true).build(),
                ImportFieldConfig.builder().field("path").label("路由路径").required(false).build(),
                ImportFieldConfig.builder().field("component").label("组件路径").required(false).build(),
                ImportFieldConfig.builder().field("perm").label("权限标识").required(false).build(),
                ImportFieldConfig.builder().field("visible").label("是否可见(显示/隐藏)").required(false).build(),
                ImportFieldConfig.builder().field("sort").label("排序").required(false).build(),
                ImportFieldConfig.builder().field("icon").label("图标").required(false).build(),
                ImportFieldConfig.builder().field("redirect").label("跳转路径").required(false).build()
        );
    }

    @Override
    public List<Map<String, Object>> getTemplateSampleData() {
        return List.of(Map.of(
                "name", "用户管理",
                "parentId", "0",
                "typeLabel", "菜单",
                "path", "/system/user",
                "component", "system/user/index",
                "perm", "sys:user:list",
                "visible", "显示",
                "sort", "1",
                "icon", "user",
                "redirect", ""
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
                    throw new IllegalArgumentException("菜单名称为空");
                }
                String typeLabel = getAsString(row, "typeLabel");
                if (CharSequenceUtil.isBlank(typeLabel)) {
                    throw new IllegalArgumentException("菜单类型为空");
                }
                MenuTypeEnum typeEnum = parseMenuType(typeLabel);
                if (typeEnum == null) {
                    throw new IllegalArgumentException("菜单类型无效(应为 菜单/目录/外链/按钮): " + typeLabel);
                }

                MenuForm form = new MenuForm();
                form.setName(name);
                form.setParentId(parseLong(row, "parentId", 0L));
                form.setType(typeEnum);
                form.setPath(getAsString(row, "path"));
                form.setComponent(getAsString(row, "component"));
                form.setPerm(getAsString(row, "perm"));
                form.setVisible(parseVisible(row, "visible", 1));
                form.setSort(parseInteger(row, "sort", 0));
                form.setIcon(getAsString(row, "icon"));
                form.setRedirect(getAsString(row, "redirect"));

                menuService.saveMenu(form);
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

    private MenuTypeEnum parseMenuType(String label) {
        for (MenuTypeEnum e : MenuTypeEnum.values()) {
            if (e.getLabel() != null && e.getLabel().equals(label)) {
                return e;
            }
        }
        return null;
    }

    private Integer parseVisible(Map<String, Object> row, String key, Integer defaultValue) {
        String v = getAsString(row, key);
        if (CharSequenceUtil.isBlank(v)) {
            return defaultValue;
        }
        if ("显示".equals(v)) return 1;
        if ("隐藏".equals(v)) return 0;
        return defaultValue;
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
