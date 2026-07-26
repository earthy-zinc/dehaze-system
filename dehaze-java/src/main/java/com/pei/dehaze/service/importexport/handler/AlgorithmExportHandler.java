package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.collection.CollUtil;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.AlgorithmStatusEnum;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.service.SysAlgorithmService;
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
 * 算法导出处理器
 * <p>树形结构导出为扁平行。仅导出元数据，不含权重文件。
 */
@Component
@RequiredArgsConstructor
public class AlgorithmExportHandler implements ExportHandler {

    private final SysAlgorithmService algorithmService;

    @Override
    public String getModule() {
        return "algorithm";
    }

    @Override
    public long estimateCount(Map<String, Object> queryParams) {
        AlgorithmQuery query = buildQuery(queryParams);
        List<AlgorithmVO> tree = algorithmService.getList(query);
        return countTree(tree);
    }

    @Override
    public void export(ExportContext ctx, ProgressCallback callback) throws Exception {
        // 通过 getDataProvider + fileGenerator 写入
    }

    @Override
    public List<ExportFieldConfig> getFieldConfigs() {
        return List.of(
                ExportFieldConfig.builder().field("name").label("算法名称").order(1).build(),
                ExportFieldConfig.builder().field("parentId").label("父算法ID").order(2).build(),
                ExportFieldConfig.builder().field("type").label("算法类型").order(3).build(),
                ExportFieldConfig.builder().field("path").label("模型文件路径").order(4).build(),
                ExportFieldConfig.builder().field("importPath").label("导入路径").order(5).build(),
                ExportFieldConfig.builder().field("description").label("描述").order(6).build(),
                ExportFieldConfig.builder().field("version").label("版本").order(7).build(),
                ExportFieldConfig.builder().field("statusLabel").label("状态").order(8).build(),
                ExportFieldConfig.builder().field("size").label("大小").order(9).build(),
                ExportFieldConfig.builder().field("flops").label("FLOPs").order(10).build(),
                ExportFieldConfig.builder().field("params").label("参数量").order(11).build()
        );
    }

    @Override
    public ExportDataProvider getDataProvider(ExportContext ctx) {
        List<ExportFieldConfig> fields = ctx.getSelectedFields() == null || ctx.getSelectedFields().isEmpty()
                ? getFieldConfigs()
                : getFieldConfigs().stream().filter(f -> ctx.getSelectedFields().contains(f.getField())).toList();

        return new ExportDataProvider() {
            private List<AlgorithmVO> flattened;

            @Override
            public List<List<Object>> fetchBatch(int pageNum, int pageSize) {
                if (flattened == null) {
                    AlgorithmQuery query = buildQuery(ctx.getQueryParams());
                    flattened = flatten(algorithmService.getList(query));
                }
                int start = (pageNum - 1) * pageSize;
                if (start >= flattened.size()) {
                    return List.of();
                }
                int end = Math.min(start + pageSize, flattened.size());
                List<AlgorithmVO> sub = flattened.subList(start, end);
                List<List<Object>> rows = new ArrayList<>(sub.size());
                for (AlgorithmVO vo : sub) {
                    rows.add(toRow(vo, fields));
                }
                return rows;
            }
        };
    }

    private AlgorithmQuery buildQuery(Map<String, Object> params) {
        AlgorithmQuery query = new AlgorithmQuery();
        if (params != null) {
            Object keywords = params.get("keywords");
            if (keywords != null) {
                query.setKeywords(String.valueOf(keywords));
            }
        }
        return query;
    }

    private List<AlgorithmVO> flatten(List<AlgorithmVO> tree) {
        List<AlgorithmVO> list = new ArrayList<>();
        if (CollUtil.isEmpty(tree)) {
            return list;
        }
        for (AlgorithmVO node : tree) {
            list.add(node);
            if (CollUtil.isNotEmpty(node.getChildren())) {
                list.addAll(flatten(node.getChildren()));
            }
        }
        return list;
    }

    private long countTree(List<AlgorithmVO> tree) {
        if (CollUtil.isEmpty(tree)) {
            return 0;
        }
        long count = 0;
        for (AlgorithmVO node : tree) {
            count++;
            if (CollUtil.isNotEmpty(node.getChildren())) {
                count += countTree(node.getChildren());
            }
        }
        return count;
    }

    private List<Object> toRow(AlgorithmVO vo, List<ExportFieldConfig> fields) {
        List<Object> row = new ArrayList<>(fields.size());
        for (ExportFieldConfig f : fields) {
            row.add(extractField(vo, f));
        }
        return row;
    }

    private Object extractField(AlgorithmVO vo, ExportFieldConfig f) {
        return switch (f.getField()) {
            case "name" -> nullToEmpty(vo.getName());
            case "parentId" -> vo.getParentId() == null ? "" : vo.getParentId();
            case "type" -> nullToEmpty(vo.getType());
            case "path" -> nullToEmpty(vo.getPath());
            case "importPath" -> nullToEmpty(vo.getImportPath());
            case "description" -> nullToEmpty(vo.getDescription());
            case "version" -> ""; // AlgorithmVO 没有 version 字段，从 entity 获取需另行处理
            case "statusLabel" -> vo.getStatus() == null
                    ? ""
                    : IBaseEnum.getLabelByValue(vo.getStatus(), AlgorithmStatusEnum.class);
            case "size" -> nullToEmpty(vo.getSize());
            case "flops" -> nullToEmpty(vo.getFlops());
            case "params" -> nullToEmpty(vo.getParams());
            default -> "";
        };
    }

    private String nullToEmpty(String s) {
        return s == null ? "" : s;
    }
}
