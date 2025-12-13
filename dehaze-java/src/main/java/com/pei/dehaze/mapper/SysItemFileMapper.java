package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.vo.ImageUrlVO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

@Mapper
public interface SysItemFileMapper extends BaseMapper<SysItemFile> {
    /**
     * 获取当前数据项下所有图片信息
     */
    List<ImageUrlVO> listImageUrl(Long itemId);

    /**
     * 批量获取数据项下的图片信息
     */
    List<ImageUrlVO> listImageUrlByItemIds(@Param("itemIds") List<Long> itemIds);

    /**
     * 增加图片使用次数
     */
    @Update("UPDATE sys_item_file SET usage_count = COALESCE(usage_count, 0) + 1 WHERE id = #{id}")
    void incrementUsageCount(@Param("id") Long id);
}
