package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiArtifact;
import org.apache.ibatis.annotations.Mapper;

/**
 * AI 中间产物访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiArtifactMapper extends BaseMapper<SysAiArtifact> {
}
