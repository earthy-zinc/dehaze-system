package com.pei.clarity.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.clarity.model.entity.SysAiAgent;
import org.apache.ibatis.annotations.Mapper;

/**
 * AI 智能体访问层
 *
 * @author clarity
 */
@Mapper
public interface SysAiAgentMapper extends BaseMapper<SysAiAgent> {}
