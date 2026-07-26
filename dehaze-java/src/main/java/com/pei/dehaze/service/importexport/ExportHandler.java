package com.pei.dehaze.service.importexport;

import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.strategy.ProgressCallback;

import java.util.List;
import java.util.Map;

/**
 * 导出处理器接口
 * <p>各模块实现该接口，提供模块特定的数据查询和字段配置逻辑。
 * 通用策略通过 {@link ExportHandlerRegistry} 查找对应 Handler 执行。
 */
public interface ExportHandler {

    /**
     * 获取模块标识（如 user/role/dept/menu/dict/dataset/algorithm）
     */
    String getModule();

    /**
     * 估算导出数据总量（用于判断同步/异步、进度计算）
     * @param queryParams 查询参数
     * @return 数据条数
     */
    long estimateCount(Map<String, Object> queryParams);

    /**
     * 执行导出
     * <p>实现方需将数据写入 {@link ExportContext#getOutputStream()}，
     * 并通过 callback 上报进度、检测取消。
     * @param ctx      导出上下文
     * @param callback 进度回调
     */
    void export(ExportContext ctx, ProgressCallback callback) throws Exception;

    /**
     * 获取字段配置（静态）
     */
    List<ExportFieldConfig> getFieldConfigs();

    /**
     * 获取动态字段配置（默认返回静态配置）
     * @param queryParams 查询参数
     */
    default List<ExportFieldConfig> getDynamicFieldConfigs(Map<String, Object> queryParams) {
        return getFieldConfigs();
    }

    /**
     * 构造导出数据提供器（流式分批拉取）
     * <p>用于通用文件生成器按批次拉取数据，避免一次性加载到内存。
     * <p>仅当 {@link #useDirectExport()} 返回 false 时调用。
     * @param ctx 导出上下文
     * @return 数据提供器
     */
    ExportDataProvider getDataProvider(ExportContext ctx);

    /**
     * 是否使用直接导出模式（如 ZIP 归档类导出）
     * <p>返回 true 时，通用服务直接调用 {@link #export(ExportContext, ProgressCallback)}
     * 写入输出流，跳过表格文件生成器路径。
     * @return 默认 false（表格导出），数据集等二进制导出返回 true
     */
    default boolean useDirectExport() {
        return false;
    }
}
