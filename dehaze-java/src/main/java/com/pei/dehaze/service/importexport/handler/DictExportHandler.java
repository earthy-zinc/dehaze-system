package com.pei.dehaze.service.importexport.handler;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.query.DictPageQuery;
import com.pei.dehaze.model.vo.DictPageVO;
import com.pei.dehaze.service.SysDictService;
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
 * 字典导出处理器
 */
@Component
@RequiredArgsConstructor
public class DictExportHandler implements ExportHandler {

    private static final int PAGE_SIZE = 1000;

    private final SysDictService dictService;

    @Override
    public String getModule() {
        return "dict";
    }

    @Override
    public long estimateCount(Map<String, Object> queryParams) {
        DictPageQuery query = buildQuery(queryParams, 1, 1);
        return dictService.getDictPage(query).getTotal();
    }

    @Override
    public void export(ExportContext ctx, ProgressCallback callback) throws Exception {
        // 通过 getDataProvider + fileGenerator 写入
    }

    @Override
    public List<ExportFieldConfig> getFieldConfigs() {
        return List.of(
                ExportFieldConfig.builder().field("typeCode").label("字典类型编码").order(1).build(),
                ExportFieldConfig.builder().field("name").label("字典名称").order(2).build(),
                ExportFieldConfig.builder().field("value").label("字典值").order(3).build(),
                ExportFieldConfig.builder().field("sort").label("排序").order(4).build(),
                ExportFieldConfig.builder().field("statusLabel").label("状态").order(5).build(),
                ExportFieldConfig.builder().field("defaulted").label("是否默认").order(6).build(),
                ExportFieldConfig.builder().field("remark").label("备注").order(7).build(),
                ExportFieldConfig.builder().field("createTime").label("创建时间").order(8)
                        .dateFormat("yyyy-MM-dd HH:mm:ss").build()
        );
    }

    @Override
    public ExportDataProvider getDataProvider(ExportContext ctx) {
        List<ExportFieldConfig> fields = ctx.getSelectedFields() == null || ctx.getSelectedFields().isEmpty()
                ? getFieldConfigs()
                : getFieldConfigs().stream().filter(f -> ctx.getSelectedFields().contains(f.getField())).toList();

        return (pageNum, pageSize) -> {
            DictPageQuery query = buildQuery(ctx.getQueryParams(), pageNum, PAGE_SIZE);
            IPage<DictPageVO> page = dictService.getDictPage(query);
            if (page.getRecords() == null || page.getRecords().isEmpty()) {
                return List.of();
            }
            List<List<Object>> rows = new ArrayList<>(page.getRecords().size());
            for (DictPageVO vo : page.getRecords()) {
                rows.add(toRow(vo, fields));
            }
            return rows;
        };
    }

    private DictPageQuery buildQuery(Map<String, Object> params, int pageNum, int pageSize) {
        DictPageQuery query = new DictPageQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        if (params != null) {
            Object keywords = params.get("keywords");
            if (keywords != null) {
                query.setKeywords(String.valueOf(keywords));
            }
            Object typeCode = params.get("typeCode");
            if (typeCode != null) {
                query.setTypeCode(String.valueOf(typeCode));
            }
        }
        return query;
    }

    private List<Object> toRow(DictPageVO vo, List<ExportFieldConfig> fields) {
        List<Object> row = new ArrayList<>(fields.size());
        for (ExportFieldConfig f : fields) {
            row.add(extractField(vo, f));
        }
        return row;
    }

    private Object extractField(DictPageVO vo, ExportFieldConfig f) {
        return switch (f.getField()) {
            case "typeCode" -> nullToEmpty(vo.getTypeCode());
            case "name" -> nullToEmpty(vo.getName());
            case "value" -> nullToEmpty(vo.getValue());
            case "sort" -> vo.getSort() == null ? "" : vo.getSort();
            case "statusLabel" -> vo.getStatus() == null
                    ? ""
                    : IBaseEnum.getLabelByValue(vo.getStatus(), StatusEnum.class);
            case "defaulted" -> vo.getDefaulted() == null ? "" : (vo.getDefaulted() == 1 ? "是" : "否");
            case "remark" -> nullToEmpty(vo.getRemark());
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
