package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysPackage;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/**
 * 安装包访问层，提供绕过@TableLogic软删过滤的原生SQL查询方法。
 */
@Mapper
public interface SysPackageMapper extends BaseMapper<SysPackage> {

    /**
     * 按名称查询安装包数（含软删行，绕过@TableLogic过滤）
     *
     * @param name 安装包名称
     * @return 匹配记录数
     */
    long countByNameAll(@Param("name") String name);

    /**
     * 按名称查询安装包数（排除指定ID，含软删行）
     *
     * @param name      安装包名称
     * @param excludeId 排除的ID
     * @return 匹配记录数
     */
    long countByNameAllExcluding(@Param("name") String name, @Param("excludeId") Long excludeId);
}
