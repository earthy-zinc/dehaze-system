package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysItemFile;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Update;

import java.util.List;
import java.util.Map;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:27:05
 */
@Mapper
public interface SysDatasetMapper extends BaseMapper<SysDataset> {

    /**
     * 增加数据集使用次数
     */
    @Update("UPDATE sys_dataset SET usage_count = COALESCE(usage_count, 0) + 1 WHERE id = #{id}")
    void incrementUsageCount(@Param("id") Long id);

    /**
     * 获取数据集下所有图片文件列表（用于下载）
     *
     * @param datasetIds 数据集ID列表
     * @return 图片文件列表
     */
    List<SysItemFile> getDatasetImages(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计数据集的综合信息（图片数、总大小、清晰图、有雾图），一次查询替代4次
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 综合统计Map
     */
    Map<String, Object> countDatasetStatsSingle(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计场景类型分布
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 场景类型分布Map
     */
    List<Map<String, Object>> countSceneDistribution(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计雾霾程度分布
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 雾霾程度分布Map
     */
    List<Map<String, Object>> countHazeDistribution(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计文件格式分布（按数据集ID）
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 文件格式分布Map
     */
    List<Map<String, Object>> countFormatDistributionByDatasetIds(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计数据项总数
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 数据项总数
     */
    Long countItemsByDatasetIds(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 批量统计每个数据集的数据项数（GROUP BY dataset_id）
     */
    List<Map<String, Object>> countItemsPerDataset(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 批量统计每个数据集的综合统计信息（GROUP BY dataset_id）
     * 包括：图片数、文件总大小、清晰图片数、有雾图片数
     */
    List<Map<String, Object>> countDatasetStatsBatch(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 批量统计每个数据集的场景类型分布（GROUP BY dataset_id, scene_type）
     */
    List<Map<String, Object>> countSceneDistributionBatch(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 批量统计每个数据集的雾霾程度分布（GROUP BY dataset_id, haze_level）
     */
    List<Map<String, Object>> countHazeDistributionBatch(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 批量统计每个数据集的文件格式分布（GROUP BY dataset_id, file_type）
     */
    List<Map<String, Object>> countFormatDistributionBatch(@Param("datasetIds") List<Long> datasetIds);
}
