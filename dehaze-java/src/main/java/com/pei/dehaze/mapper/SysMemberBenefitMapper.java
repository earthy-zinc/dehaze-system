package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/**
 * 会员权益访问层，提供绕过@TableLogic软删过滤的原生SQL查询方法。
 */
@Mapper
public interface SysMemberBenefitMapper extends BaseMapper<SysMemberBenefit> {

    /**
     * 按等级编码查询会员权益数（含软删行，绕过@TableLogic过滤）
     *
     * @param levelCode 等级编码
     * @return 匹配记录数
     */
    long countByLevelCodeAll(@Param("levelCode") String levelCode);

    /**
     * 按等级编码查询会员权益数（排除指定ID，含软删行）
     *
     * @param levelCode 等级编码
     * @param excludeId 排除的ID
     * @return 匹配记录数
     */
    long countByLevelCodeAllExcluding(@Param("levelCode") String levelCode, @Param("excludeId") Long excludeId);
}
