package com.pei.dehaze.service.importexport.model;

import java.util.List;

/**
 * 导出数据提供器
 * <p>由 {@link com.pei.dehaze.service.importexport.ExportHandler} 实现，
 * 供 {@link com.pei.dehaze.service.importexport.ImportExportFileGenerator} 分批拉取数据。
 */
@FunctionalInterface
public interface ExportDataProvider {

    /**
     * 拉取指定批次的数据
     * @param pageNum  页码（从 1 开始）
     * @param pageSize 每页大小
     * @return 当前批次的数据行（每行为字段值列表，顺序与 FieldConfig 一致）；空列表表示无更多数据
     */
    List<List<Object>> fetchBatch(int pageNum, int pageSize);
}
