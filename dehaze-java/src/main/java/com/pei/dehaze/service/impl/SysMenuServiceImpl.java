package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.MenuTypeEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.converter.MenuConverter;
import com.pei.dehaze.mapper.SysMenuMapper;
import com.pei.dehaze.model.bo.RouteBO;
import com.pei.dehaze.model.entity.SysMenu;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysRoleMenu;
import com.pei.dehaze.model.form.MenuForm;
import com.pei.dehaze.model.query.MenuQuery;
import com.pei.dehaze.model.vo.MenuVO;
import com.pei.dehaze.model.vo.RouteVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysMenuService;
import com.pei.dehaze.service.SysRoleMenuService;
import com.pei.dehaze.service.SysRoleService;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.StringUtils;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 菜单业务实现类
 *
 * @author earthyzinc
 * @since 2020/11/06
 */
@Service
@RequiredArgsConstructor
public class SysMenuServiceImpl extends ServiceImpl<SysMenuMapper, SysMenu> implements SysMenuService {

    private final MenuConverter menuConverter;

    private final SysRoleMenuService roleMenuService;

    private final SysRoleService roleService;

    private final StringRedisTemplate stringRedisTemplate;

    private static final String MENU_ROUTES_KEY = "menu:routes";

    private static final String MENU_OPTIONS_KEY = "menu:options";

    private static final Duration MENU_CACHE_TTL = Duration.ofHours(1);


    /**
     * 菜单列表
     *
     * @param queryParams {@link MenuQuery}
     */
    @Override
    public List<MenuVO> listMenus(MenuQuery queryParams) {
        List<SysMenu> menus = this.list(new LambdaQueryWrapper<SysMenu>()
                .like(CharSequenceUtil.isNotBlank(queryParams.getKeywords()), SysMenu::getName, queryParams.getKeywords())
                .orderByAsc(SysMenu::getSort)
        );
        List<Long> rootIds = TreeDataUtils.findRootIds(menus, SysMenu::getId, SysMenu::getParentId);

        // 构建 parentId -> children Map，避免递归内 O(n) 过滤
        Map<Long, List<SysMenu>> parentToChildrenMap = menus.stream()
                .collect(Collectors.groupingBy(SysMenu::getParentId));

        // 递归函数来构建菜单树
        return rootIds.stream()
                .flatMap(rootId -> buildMenuTree(rootId, parentToChildrenMap).stream())
                .toList();
    }

    /**
     * 新增/修改菜单
     */
    @Override
    public boolean saveMenu(MenuForm menuForm) {

        // 修改时检查菜单是否存在
        if (menuForm.getId() != null) {
            SysMenu existingMenu = this.getById(menuForm.getId());
            if (existingMenu == null) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在");
            }
        }

        MenuTypeEnum menuType = menuForm.getType();

        if (menuType == MenuTypeEnum.CATALOG) {  // 如果是目录
            String path = menuForm.getPath();
            if (menuForm.getParentId() == 0 && !path.startsWith("/")) {
                menuForm.setPath("/" + path); // 一级目录需以 / 开头
            }
            menuForm.setComponent("Layout");
        } else if (menuType == MenuTypeEnum.EXTLINK) {   // 如果是外链

            menuForm.setComponent(null);
        }

        SysMenu entity = menuConverter.form2Entity(menuForm);
        String treePath = generateMenuTreePath(menuForm.getParentId());
        entity.setTreePath(treePath);

        boolean isNew = menuForm.getId() == null;
        boolean result = this.saveOrUpdate(entity);
        if (result) {
            if (isNew) {
                // 新增菜单默认分配给超级管理员角色
                SysRole rootRole = roleService.getOne(new LambdaQueryWrapper<SysRole>()
                        .eq(SysRole::getCode, SystemConstants.ROOT_ROLE_CODE));
                if (rootRole != null) {
                    roleMenuService.save(new SysRoleMenu(rootRole.getId(), entity.getId()));
                }
            }
            evictMenuCache();
            roleMenuService.refreshRolePermsCache();
        }
        return result;
    }

    /**
     * 菜单下拉数据
     */
    @Override
    public List<Option<Long>> listMenuOptions() {
        String cached = stringRedisTemplate.opsForValue().get(MENU_OPTIONS_KEY);
        if (cached != null) {
            return (List<Option<Long>>) (List<?>) JSONUtil.parseArray(cached).toList(Option.class);
        }
        List<SysMenu> menuList = this.list(new LambdaQueryWrapper<SysMenu>()
                .orderByAsc(SysMenu::getSort));
        // 构建 parentId -> children Map，避免 O(n²) 递归
        Map<Long, List<SysMenu>> parentToChildrenMap = menuList.stream()
                .collect(Collectors.groupingBy(SysMenu::getParentId));
        List<Option<Long>> options = buildMenuOptions(SystemConstants.ROOT_NODE_ID, parentToChildrenMap);
        stringRedisTemplate.opsForValue().set(MENU_OPTIONS_KEY, JSONUtil.toJsonStr(options), MENU_CACHE_TTL);
        return options;
    }

    /**
     * 递归生成菜单下拉层级列表
     *
     * @param parentId           父级ID
     * @param parentToChildrenMap 父级ID -> 子菜单列表 的Map（预先分组，O(1)查找）
     * @return 菜单下拉列表
     */
    private List<Option<Long>> buildMenuOptions(Long parentId, Map<Long, List<SysMenu>> parentToChildrenMap) {
        List<Option<Long>> menuOptions = new ArrayList<>();

        List<SysMenu> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        for (SysMenu menu : children) {
            Option<Long> option = new Option<>(menu.getId(), menu.getName());
            List<Option<Long>> subMenuOptions = buildMenuOptions(menu.getId(), parentToChildrenMap);
            if (!subMenuOptions.isEmpty()) {
                option.setChildren(subMenuOptions);
            }
            menuOptions.add(option);
        }

        return menuOptions;
    }

    /**
     * 获取路由列表
     */
    @Override
    public List<RouteVO> listRoutes() {
        String cached = stringRedisTemplate.opsForValue().get(MENU_ROUTES_KEY);
        if (cached != null) {
            return JSONUtil.parseArray(cached).toList(RouteVO.class);
        }
        List<RouteBO> menuList = this.baseMapper.listRoutes();
        List<RouteVO> routes = buildRoutes(SystemConstants.ROOT_NODE_ID, menuList);
        stringRedisTemplate.opsForValue().set(MENU_ROUTES_KEY, JSONUtil.toJsonStr(routes), MENU_CACHE_TTL);
        return routes;
    }

    /**
     * 递归生成菜单路由层级列表
     *
     * @param parentId 父级ID
     * @param menuList 菜单列表
     * @return 路由层级列表
     */
    private List<RouteVO> buildRoutes(Long parentId, List<RouteBO> menuList) {
        List<RouteVO> routeList = new ArrayList<>();

        for (RouteBO menu : menuList) {
            if (menu.getParentId().equals(parentId)) {
                RouteVO routeVO = toRouteVo(menu);
                List<RouteVO> children = buildRoutes(menu.getId(), menuList);
                if (!children.isEmpty()) {
                    routeVO.setChildren(children);
                }
                routeList.add(routeVO);
            }
        }

        return routeList;
    }

    /**
     * 递归生成菜单列表
     *
     * @param parentId           父级ID
     * @param parentToChildrenMap 父级ID -> 子菜单列表 的Map（预先分组，O(1)查找）
     * @return 菜单列表
     */
    private List<MenuVO> buildMenuTree(Long parentId, Map<Long, List<SysMenu>> parentToChildrenMap) {
        List<MenuVO> menuList = new ArrayList<>();
        List<SysMenu> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        for (SysMenu menu : children) {
            MenuVO menuVO = menuConverter.entity2Vo(menu);
            List<MenuVO> subMenuList = buildMenuTree(menu.getId(), parentToChildrenMap);
            if (!subMenuList.isEmpty()) {
                menuVO.setChildren(subMenuList);
            }
            menuList.add(menuVO);
        }
        return menuList;
    }

    /**
     * 根据RouteBO创建RouteVO
     */
    private RouteVO toRouteVo(RouteBO routeBO) {
        RouteVO routeVO = new RouteVO();
        String routeName = StringUtils.capitalize(CharSequenceUtil.toCamelCase(routeBO.getPath(), '-'));  // 路由 name 需要驼峰，首字母大写
        routeVO.setName(routeName); // 根据name路由跳转 this.$router.push({name:xxx})
        routeVO.setPath(routeBO.getPath()); // 根据path路由跳转 this.$router.push({path:xxx})
        routeVO.setRedirect(routeBO.getRedirect());
        routeVO.setComponent(routeBO.getComponent());

        RouteVO.Meta meta = new RouteVO.Meta();
        meta.setTitle(routeBO.getName());
        meta.setIcon(routeBO.getIcon());
        meta.setRoles(routeBO.getRoles());
        meta.setHidden(StatusEnum.DISABLE.getValue().equals(routeBO.getVisible()));
        // 【菜单】是否开启页面缓存
        if (MenuTypeEnum.MENU.equals(routeBO.getType())
                && ObjectUtil.equals(routeBO.getKeepAlive(), 1)) {
            meta.setKeepAlive(true);
        }
        // 【目录】只有一个子路由是否始终显示
        if (MenuTypeEnum.CATALOG.equals(routeBO.getType())
                && ObjectUtil.equals(routeBO.getAlwaysShow(), 1)) {
            meta.setAlwaysShow(true);
        }

        routeVO.setMeta(meta);
        return routeVO;
    }

    /**
     * 菜单路径生成
     *
     * @param parentId 父ID
     * @return 父节点路径以英文逗号(, )分割，eg: 1,2,3
     */
    private String generateMenuTreePath(Long parentId) {
        if (SystemConstants.ROOT_NODE_ID.equals(parentId)) {
            return String.valueOf(parentId);
        }
        SysMenu parent = this.getById(parentId);
        if (parent == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "父级菜单不存在");
        }
        return parent.getTreePath() + "," + parent.getId();
    }


    /**
     * 修改菜单显示状态
     *
     * @param menuId  菜单ID
     * @param visible 是否显示(1->显示；2->隐藏)
     * @return 是否修改成功
     */
    @Override
    public boolean updateMenuVisible(Long menuId, Integer visible) {
        Long currentUserId = SecurityUtils.getUserId();
        boolean result = this.update(new LambdaUpdateWrapper<SysMenu>()
                .eq(SysMenu::getId, menuId)
                .set(SysMenu::getVisible, visible)
                .set(SysMenu::getUpdateBy, currentUserId)
        );
        if (result) {
            evictMenuCache();
        }
        return result;
    }

    /**
     * 获取角色权限(Code)集合
     *
     * @param roles 角色Code集合
     * @return 权限集合
     */
    @Override
    public Set<String> listRolePerms(Set<String> roles) {
        return this.baseMapper.listRolePerms(roles);
    }

    /**
     * 获取菜单表单数据
     *
     * @param id 菜单ID
     * @return 菜单表单数据
     */
    @Override
    public MenuForm getMenuForm(Long id) {
        SysMenu entity = this.getById(id);
        if (entity == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在");
        }
        return menuConverter.entity2Form(entity);
    }

    /**
     * 批量删除菜单（级联删除子孙菜单，并清理角色-菜单关联）
     *
     * @param ids 菜单ID集合
     * @return 是否删除成功
     */
    @Override
    public boolean deleteMenu(List<Long> ids) {
        if (ids == null || ids.isEmpty()) {
            return true;
        }

        // 校验所有传入的菜单ID都存在
        long existCount = this.count(new LambdaQueryWrapper<SysMenu>().in(SysMenu::getId, ids));
        if (existCount != ids.size()) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在");
        }

        // 一次性查询所有待删除菜单ID（传入ID + 子孙），合并去重
        // 条件：id IN (ids) OR tree_path LIKE '%,id,%'（对每个 id 做 OR）
        LambdaQueryWrapper<SysMenu> wrapper = new LambdaQueryWrapper<SysMenu>()
                .in(SysMenu::getId, ids);
        for (Long id : ids) {
            wrapper.or().apply("CONCAT (',',tree_path,',') LIKE CONCAT('%,',{0},',%')", id);
        }
        List<Long> menuIds = this.list(wrapper).stream()
                .map(SysMenu::getId).distinct().toList();

        if (menuIds.isEmpty()) {
            return true;
        }

        // 删除角色-菜单关联
        roleMenuService.remove(new LambdaQueryWrapper<SysRoleMenu>()
                .in(SysRoleMenu::getMenuId, menuIds));

        // 删除菜单
        boolean result = this.removeByIds(menuIds);

        // 刷新角色权限缓存
        if (result) {
            evictMenuCache();
            roleMenuService.refreshRolePermsCache();
        }
        return result;
    }

    /**
     * 清除菜单路由和选项缓存
     */
    private void evictMenuCache() {
        stringRedisTemplate.delete(MENU_ROUTES_KEY);
        stringRedisTemplate.delete(MENU_OPTIONS_KEY);
    }

}
