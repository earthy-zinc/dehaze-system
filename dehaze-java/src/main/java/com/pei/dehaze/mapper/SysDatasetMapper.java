package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysItemFile;
import org.apache.ibatis.annotations.MapKey;
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
     * 统计文件格式分布
     *
     * @param itemIds 数据项ID列表
     * @return 文件格式分布List，每项包含 key(格式) 和 value(数量)
     */
    @MapKey("key")
    List<Map<String, Object>> countFormatDistribution(@Param("itemIds") List<Long> itemIds);

    /**
     * 获取数据集下所有图片文件列表（用于下载）
     *
     * @param datasetIds 数据集ID列表
     * @return 图片文件列表
     */
    List<SysItemFile> getDatasetImages(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计数据集图片总数
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 图片总数
     */
    Long countImagesByDatasetIds(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计场景类型分布
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 场景类型分布Map
     */
    @MapKey("scene_type")
    List<Map<String, Object>> countSceneDistribution(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计雾霾程度分布
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 雾霾程度分布Map
     */
    @MapKey("haze_level")
    List<Map<String, Object>> countHazeDistribution(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计文件格式分布（按数据集ID）
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 文件格式分布Map
     */
    @MapKey("file_type")
    List<Map<String, Object>> countFormatDistributionByDatasetIds(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计数据项总数
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 数据项总数
     */
    Long countItemsByDatasetIds(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计文件总大小（字节）
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 文件总大小
     */
    Long countTotalSizeByDatasetIds(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计清晰图片数量
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 清晰图片数量
     */
    Long countClearImagesByDatasetIds(@Param("datasetIds") List<Long> datasetIds);

    /**
     * 统计有雾图片数量
     *
     * @param datasetIds 数据集ID列表（叶子节点）
     * @return 有雾图片数量
     */
    Long countHazyImagesByDatasetIds(@Param("datasetIds") List<Long> datasetIds);
}
