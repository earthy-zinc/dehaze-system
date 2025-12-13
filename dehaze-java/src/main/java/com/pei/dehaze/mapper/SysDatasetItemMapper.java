package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.vo.DatasetItemVO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface SysDatasetItemMapper extends BaseMapper<SysDatasetItem> {

    /**
     * 搜索图片（支持多条件筛选和相关度排序）
     */
    List<DatasetItemVO> searchImages(
            Page<DatasetItemVO> page,
            @Param("leafIds") List<Long> leafIds,
            @Param("keywords") String keywords,
            @Param("sceneType") String sceneType,
            @Param("hazeLevel") String hazeLevel,
            @Param("minWidth") Integer minWidth,
            @Param("maxWidth") Integer maxWidth,
            @Param("minHeight") Integer minHeight,
            @Param("maxHeight") Integer maxHeight,
            @Param("minSize") Long minSize,
            @Param("maxSize") Long maxSize,
            @Param("sortBy") String sortBy,
            @Param("sortOrder") String sortOrder
    );
}
