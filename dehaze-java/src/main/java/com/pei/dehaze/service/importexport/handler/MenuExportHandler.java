package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.collection.CollUtil;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.MenuTypeEnum;
import com.pei.dehaze.model.query.MenuQuery;
import com.pei.dehaze.model.vo.MenuVO;
import com.pei.dehaze.service.SysMenuService;
import com.pei.dehaze.service.importexport.ExportHandler;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.strategy.ProgressCallback;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 菜单导出处理器
 * <p>树形结构导出为扁平行。
 */
@Component
@RequiredArgsConstructor
public class MenuExportHandler implements ExportHandler {

    private final SysMenuService menuService;

    @Override
    public String getModule() {
        return "menu";
    }

    @Override
    public long estimateCount(Map<String, Object> queryParams) {
        MenuQuery query = buildQuery(queryParams);
        List<MenuVO> tree = menuService.listMenus(query);
        return countTree(tree);
    }

    @Override
    public void export(ExportContext ctx, ProgressCallback callback) throws Exception {
        // 通过 getDataProvider + fileGenerator 写入
    }

    @Override
    public List<ExportFieldConfig> getFieldConfigs() {
        return List.of(
                ExportFieldConfig.builder().field("name").label("菜单名称").order(1).build(),
                ExportFieldConfig.builder().field("parentId").label("父菜单ID").order(2).build(),
                ExportFieldConfig.builder().field("typeLabel").label("类型").order(3).build(),
                ExportFieldConfig.builder().field("path").label("路由路径").order(4).build(),
                ExportFieldConfig.builder().field("component").label("组件路径").order(5).build(),
                ExportFieldConfig.builder().field("perm").label("权限标识").order(6).build(),
                ExportFieldConfig.builder().field("visible").label("是否可见").order(7).build(),
                ExportFieldConfig.builder().field("sort").label("排序").order(8).build(),
                ExportFieldConfig.builder().field("icon").label("图标").order(9).build(),
                ExportFieldConfig.builder().field("redirect").label("跳转路径").order(10).build()
        );
    }

    @Override
    public ExportDataProvider getDataProvider(ExportContext ctx) {
        List<ExportFieldConfig> fields = ctx.getSelectedFields() == null || ctx.getSelectedFields().isEmpty()
                ? getFieldConfigs()
                : getFieldConfigs().stream().filter(f -> ctx.getSelectedFields().contains(f.getField())).toList();

        return new ExportDataProvider() {
            private List<MenuVO> flattened;

            @Override
            public List<List<Object>> fetchBatch(int pageNum, int pageSize) {
                if (flattened == null) {
                    MenuQuery query = buildQuery(ctx.getQueryParams());
                    flattened = flatten(menuService.listMenus(query));
                }
                int start = (pageNum - 1) * pageSize;
                if (start >= flattened.size()) {
                    return List.of();
                }
                int end = Math.min(start + pageSize, flattened.size());
                List<MenuVO> sub = flattened.subList(start, end);
                List<List<Object>> rows = new ArrayList<>(sub.size());
                for (MenuVO vo : sub) {
                    rows.add(toRow(vo, fields));
                }
                return rows;
            }
        };
    }

    private MenuQuery buildQuery(Map<String, Object> params) {
        MenuQuery query = new MenuQuery();
        if (params != null) {
            Object keywords = params.get("keywords");
            if (keywords != null) {
                query.setKeywords(String.valueOf(keywords));
            }
            Object status = params.get("status");
            if (status != null && !"".equals(String.valueOf(status))) {
                query.setStatus(Integer.valueOf(String.valueOf(status)));
            }
        }
        return query;
    }

    private List<MenuVO> flatten(List<MenuVO> tree) {
        List<MenuVO> list = new ArrayList<>();
        if (CollUtil.isEmpty(tree)) {
            return list;
        }
        for (MenuVO node : tree) {
            list.add(node);
            if (CollUtil.isNotEmpty(node.getChildren())) {
                list.addAll(flatten(node.getChildren()));
            }
        }
        return list;
    }

    private long countTree(List<MenuVO> tree) {
        if (CollUtil.isEmpty(tree)) {
            return 0;
        }
        long count = 0;
        for (MenuVO node : tree) {
            count++;
            if (CollUtil.isNotEmpty(node.getChildren())) {
                count += countTree(node.getChildren());
            }
        }
        return count;
    }

    private List<Object> toRow(MenuVO vo, List<ExportFieldConfig> fields) {
        List<Object> row = new ArrayList<>(fields.size());
        for (ExportFieldConfig f : fields) {
            row.add(extractField(vo, f));
        }
        return row;
    }

    private Object extractField(MenuVO vo, ExportFieldConfig f) {
        return switch (f.getField()) {
            case "name" -> nullToEmpty(vo.getName());
            case "parentId" -> vo.getParentId() == null ? "" : vo.getParentId();
            case "typeLabel" -> vo.getType() == null
                    ? ""
                    : IBaseEnum.getLabelByValue(vo.getType().getValue(), MenuTypeEnum.class);
            case "path" -> nullToEmpty(vo.getPath());
            case "component" -> nullToEmpty(vo.getComponent());
            case "perm" -> nullToEmpty(vo.getPerm());
            case "visible" -> vo.getVisible() == null ? "" : (vo.getVisible() == 1 ? "显示" : "隐藏");
            case "sort" -> vo.getSort() == null ? "" : vo.getSort();
            case "icon" -> nullToEmpty(vo.getIcon());
            case "redirect" -> nullToEmpty(vo.getRedirect());
            default -> "";
        };
    }

    private String nullToEmpty(String s) {
        return s == null ? "" : s;
    }
}
