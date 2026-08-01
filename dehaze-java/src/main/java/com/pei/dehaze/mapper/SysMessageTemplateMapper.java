package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysMessageTemplate;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/**
 * 消息模板访问层，提供绕过@TableLogic软删过滤的原生SQL查询方法。
 */
@Mapper
public interface SysMessageTemplateMapper extends BaseMapper<SysMessageTemplate> {

    /**
     * 按编码查询消息模板数（含软删行，绕过@TableLogic过滤）
     *
     * @param code 消息模板编码
     * @return 匹配记录数
     */
    long countByCodeAll(@Param("code") String code);

    /**
     * 按编码查询消息模板数（排除指定ID，含软删行）
     *
     * @param code      消息模板编码
     * @param excludeId 排除的ID
     * @return 匹配记录数
     */
    long countByCodeAllExcluding(@Param("code") String code, @Param("excludeId") Long excludeId);
}
