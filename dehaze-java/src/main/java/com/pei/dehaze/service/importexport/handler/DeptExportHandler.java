package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.collection.CollUtil;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.query.DeptQuery;
import com.pei.dehaze.model.vo.DeptVO;
import com.pei.dehaze.service.SysDeptService;
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
 * 部门导出处理器
 * <p>树形结构导出为扁平行，按树遍历顺序输出。
 */
@Component
@RequiredArgsConstructor
public class DeptExportHandler implements ExportHandler {

    private static final int PAGE_SIZE = 1000;

    private final SysDeptService deptService;

    @Override
    public String getModule() {
        return "dept";
    }

    @Override
    public long estimateCount(Map<String, Object> queryParams) {
        DeptQuery query = buildQuery(queryParams);
        List<DeptVO> tree = deptService.listDepartments(query);
        return countTree(tree);
    }

    @Override
    public void export(ExportContext ctx, ProgressCallback callback) throws Exception {
        // 通过 getDataProvider + fileGenerator 写入
    }

    @Override
    public List<ExportFieldConfig> getFieldConfigs() {
        return List.of(
                ExportFieldConfig.builder().field("name").label("部门名称").order(1).build(),
                ExportFieldConfig.builder().field("parentId").label("父部门ID").order(2).build(),
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

        // 树形数据一次性加载，内部按批次切片
        return new ExportDataProvider() {
            private List<DeptVO> flattened;
            private int loaded = 0;

            @Override
            public List<List<Object>> fetchBatch(int pageNum, int pageSize) {
                if (flattened == null) {
                    DeptQuery query = buildQuery(ctx.getQueryParams());
                    flattened = flatten(deptService.listDepartments(query));
                }
                int start = (pageNum - 1) * pageSize;
                if (start >= flattened.size()) {
                    return List.of();
                }
                int end = Math.min(start + pageSize, flattened.size());
                List<DeptVO> sub = flattened.subList(start, end);
                List<List<Object>> rows = new ArrayList<>(sub.size());
                for (DeptVO vo : sub) {
                    rows.add(toRow(vo, fields));
                }
                loaded = end;
                return rows;
            }
        };
    }

    private DeptQuery buildQuery(Map<String, Object> params) {
        DeptQuery query = new DeptQuery();
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

    private List<DeptVO> flatten(List<DeptVO> tree) {
        List<DeptVO> list = new ArrayList<>();
        if (CollUtil.isEmpty(tree)) {
            return list;
        }
        for (DeptVO node : tree) {
            list.add(node);
            if (CollUtil.isNotEmpty(node.getChildren())) {
                list.addAll(flatten(node.getChildren()));
            }
        }
        return list;
    }

    private long countTree(List<DeptVO> tree) {
        if (CollUtil.isEmpty(tree)) {
            return 0;
        }
        long count = 0;
        for (DeptVO node : tree) {
            count++;
            if (CollUtil.isNotEmpty(node.getChildren())) {
                count += countTree(node.getChildren());
            }
        }
        return count;
    }

    private List<Object> toRow(DeptVO vo, List<ExportFieldConfig> fields) {
        List<Object> row = new ArrayList<>(fields.size());
        for (ExportFieldConfig f : fields) {
            row.add(extractField(vo, f));
        }
        return row;
    }

    private Object extractField(DeptVO vo, ExportFieldConfig f) {
        return switch (f.getField()) {
            case "name" -> nullToEmpty(vo.getName());
            case "parentId" -> vo.getParentId() == null ? "" : vo.getParentId();
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
