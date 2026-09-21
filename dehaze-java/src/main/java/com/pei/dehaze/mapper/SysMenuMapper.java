package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.read.RouteRead;
import com.pei.dehaze.model.entity.SysMenu;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Set;

/**
 * 菜单持久接口层
 *
 * @author earthyzinc
 * @since 2022/1/24
 */
@Mapper
public interface SysMenuMapper extends BaseMapper<SysMenu> {

    List<RouteRead> listRoutes();

    /**
     * 获取角色权限集合
     */
    Set<String> listRolePerms(Set<String> roles);

    /**
     * 同级菜单名称计数（含软删行，对齐 python exists_by_name 唯一性口径）
     */
    long countSameNameIncludeDeleted(@Param("parentId") Long parentId, @Param("name") String name,
            @Param("excludeId") Long excludeId);

    /**
     * 权限标识计数（含软删行，全局唯一，对齐 python exists_by_perm）
     */
    long countByPermIncludeDeleted(@Param("perm") String perm, @Param("excludeId") Long excludeId);
}
