package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysDict;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/**
 * 字典项访问层，提供绕过@TableLogic软删过滤的原生SQL查询方法。
 */
@Mapper
public interface SysDictMapper extends BaseMapper<SysDict> {

    /**
     * 按type_code+value查询字典项数（含软删行，绕过@TableLogic过滤）
     *
     * @param typeCode 字典类型编码
     * @param value    字典值
     * @return 匹配记录数
     */
    long countByTypeCodeAndValueAll(@Param("typeCode") String typeCode,
                                    @Param("value") String value);

    /**
     * 按type_code+value查询字典项数（排除指定ID，含软删行）
     *
     * @param typeCode  字典类型编码
     * @param value     字典值
     * @param excludeId 排除的ID
     * @return 匹配记录数
     */
    long countByTypeCodeAndValueAllExcluding(@Param("typeCode") String typeCode,
                                             @Param("value") String value,
                                             @Param("excludeId") Long excludeId);
}




