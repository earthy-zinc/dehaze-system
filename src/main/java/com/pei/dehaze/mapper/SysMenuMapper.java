package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.bo.RouteBO;
import com.pei.dehaze.model.entity.SysMenu;
import org.apache.ibatis.annotations.Mapper;

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

    List<RouteBO> listRoutes();

    /**
     * 获取角色权限集合
     */
    Set<String> listRolePerms(Set<String> roles);
}
