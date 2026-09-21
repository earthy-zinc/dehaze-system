package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiAgent;
import org.apache.ibatis.annotations.Mapper;

/**
 * AI 智能体访问层
 *
 * @author dehaze
 */
@Mapper
public interface SysAiAgentMapper extends BaseMapper<SysAiAgent> {
}
