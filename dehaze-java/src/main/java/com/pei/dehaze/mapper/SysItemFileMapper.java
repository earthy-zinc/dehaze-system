package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.vo.ImageUrlVO;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface SysItemFileMapper extends BaseMapper<SysItemFile> {
    /**
     * 获取当前数据项下所有图片信息
     */
    List<ImageUrlVO> listImageUrl(Long itemId);
}
