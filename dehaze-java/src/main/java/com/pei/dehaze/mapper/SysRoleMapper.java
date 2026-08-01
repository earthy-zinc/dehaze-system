package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysRole;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.Set;

@Mapper
public interface SysRoleMapper extends BaseMapper<SysRole> {


    /**
     * 获取最大范围的数据权限
     *
     * @param roles
     * @return
     */
    Integer getMaximumDataScope(Set<String> roles);

    /**
     * 按编码查询角色数（含软删行，绕过@TableLogic过滤）
     *
     * @param code 角色编码
     * @return 匹配记录数
     */
    long countByCodeAll(@Param("code") String code);

    /**
     * 按名称或编码查询角色数（含软删行，绕过@TableLogic过滤）
     *
     * @param name 角色名称
     * @param code 角色编码
     * @param excludeId 排除的ID（更新时传自身ID）
     * @return 匹配记录数
     */
    long countByNameOrCodeAll(@Param("name") String name,
                              @Param("code") String code,
                              @Param("excludeId") Long excludeId);
}
