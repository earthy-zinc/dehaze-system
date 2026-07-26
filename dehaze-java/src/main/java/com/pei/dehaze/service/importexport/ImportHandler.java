package com.pei.dehaze.service.importexport;

import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;

import java.util.List;
import java.util.Map;

/**
 * 导入处理器接口
 * <p>各模块实现该接口，提供模块特定的字段配置、数据校验和批量插入逻辑。
 */
public interface ImportHandler {

    /**
     * 获取模块标识
     */
    String getModule();

    /**
     * 获取字段配置（静态）
     */
    List<ImportFieldConfig> getFieldConfigs();

    /**
     * 获取动态字段配置（默认返回静态配置）
     */
    default List<ImportFieldConfig> getDynamicFieldConfigs() {
        return getFieldConfigs();
    }

    /**
     * 批量导入数据
     * @param rows     解析后的数据行（key=字段名，value=字段值）
     * @param options  导入选项
     * @param callback 进度回调
     * @return 导入结果
     */
    ImportResult importBatch(List<Map<String, Object>> rows, ImportOptions options, ProgressCallback callback);

    /**
     * 获取模板示例数据（用于生成带示例的导入模板）
     * @return 示例数据行
     */
    default List<Map<String, Object>> getTemplateSampleData() {
        return List.of();
    }
}
