package com.pei.dehaze.service.importexport.handler;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.form.RoleForm;
import com.pei.dehaze.model.query.RolePageQuery;
import com.pei.dehaze.model.vo.RolePageVO;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.importexport.ExportHandler;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.strategy.ProgressCallback;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 角色导出处理器
 */
@Component
@RequiredArgsConstructor
public class RoleExportHandler implements ExportHandler {

    private static final int PAGE_SIZE = 1000;

    private final SysRoleService roleService;

    @Override
    public String getModule() {
        return "role";
    }

    @Override
    public long estimateCount(Map<String, Object> queryParams) {
        RolePageQuery query = buildQuery(queryParams, 1, 1);
        return roleService.getRolePage(query).getTotal();
    }

    @Override
    public void export(ExportContext ctx, ProgressCallback callback) throws Exception {
        // 通过 getDataProvider + fileGenerator 写入
    }

    @Override
    public List<ExportFieldConfig> getFieldConfigs() {
        return List.of(
                ExportFieldConfig.builder().field("name").label("角色名称").order(1).build(),
                ExportFieldConfig.builder().field("code").label("角色编码").order(2).build(),
                ExportFieldConfig.builder().field("sort").label("排序").order(3).build(),
                ExportFieldConfig.builder().field("statusLabel").label("状态").order(4).build(),
                ExportFieldConfig.builder().field("createTime").label("创建时间").order(5)
                        .dateFormat("yyyy-MM-dd HH:mm:ss").build()
        );
    }

    @Override
    public ExportDataProvider getDataProvider(ExportContext ctx) {
        List<ExportFieldConfig> fields = ctx.getSelectedFields() == null || ctx.getSelectedFields().isEmpty()
                ? getFieldConfigs()
                : getFieldConfigs().stream().filter(f -> ctx.getSelectedFields().contains(f.getField())).toList();

        return (pageNum, pageSize) -> {
            RolePageQuery query = buildQuery(ctx.getQueryParams(), pageNum, PAGE_SIZE);
            IPage<RolePageVO> page = roleService.getRolePage(query);
            if (page.getRecords() == null || page.getRecords().isEmpty()) {
                return List.of();
            }
            List<List<Object>> rows = new ArrayList<>(page.getRecords().size());
            for (RolePageVO vo : page.getRecords()) {
                rows.add(toRow(vo, fields));
            }
            return rows;
        };
    }

    private RolePageQuery buildQuery(Map<String, Object> params, int pageNum, int pageSize) {
        RolePageQuery query = new RolePageQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        if (params != null) {
            Object keywords = params.get("keywords");
            if (keywords != null) {
                query.setKeywords(String.valueOf(keywords));
            }
        }
        return query;
    }

    private List<Object> toRow(RolePageVO vo, List<ExportFieldConfig> fields) {
        List<Object> row = new ArrayList<>(fields.size());
        for (ExportFieldConfig f : fields) {
            row.add(extractField(vo, f));
        }
        return row;
    }

    private Object extractField(RolePageVO vo, ExportFieldConfig f) {
        return switch (f.getField()) {
            case "name" -> nullToEmpty(vo.getName());
            case "code" -> nullToEmpty(vo.getCode());
            case "sort" -> vo.getSort() == null ? "" : vo.getSort();
            case "statusLabel" -> vo.getStatus() == null
                    ? ""
                    : IBaseEnum.getLabelByValue(vo.getStatus(), StatusEnum.class);
            case "createTime" -> vo.getCreateTime() == null
                    ? ""
                    : new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(java.sql.Timestamp.valueOf(vo.getCreateTime()));
            default -> "";
        };
    }

    private String nullToEmpty(String s) {
        return s == null ? "" : s;
    }
}
