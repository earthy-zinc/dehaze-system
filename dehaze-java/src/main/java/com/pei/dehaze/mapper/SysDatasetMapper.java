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
     * 统计文件格式分布
     */
    Map<String, Long> countFormatDistribution(@Param("itemIds") List<Long> itemIds);

    /**
     * 获取数据集下所有图片文件列表（用于下载）
     *
     * @param datasetIds 数据集ID列表
     * @return 图片文件列表
     */
    List<SysItemFile> getDatasetImages(@Param("datasetIds") List<Long> datasetIds);
}
