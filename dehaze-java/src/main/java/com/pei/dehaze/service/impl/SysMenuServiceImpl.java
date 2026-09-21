package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.MenuTypeEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.converter.MenuConverter;
import com.pei.dehaze.mapper.SysMenuMapper;
import com.pei.dehaze.mapper.SysRoleMapper;
import com.pei.dehaze.model.read.RouteRead;
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
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.StringUtils;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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

    private final SysRoleMapper roleMapper;

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
        // 条件式 eq 的 val 参数总会被求值，必须先完成枚举转换再决定是否追加条件，
        // 避免对 null/非法 type 调用转换（曾因此对不带 type 的请求全量 NPE 500）
        MenuTypeEnum typeEnum = IBaseEnum.getEnumByValue(queryParams.getType(), MenuTypeEnum.class);
        List<SysMenu> menus = this.list(new LambdaQueryWrapper<SysMenu>()
                .like(CharSequenceUtil.isNotBlank(queryParams.getKeywords()), SysMenu::getName, queryParams.getKeywords())
                .like(CharSequenceUtil.isNotBlank(queryParams.getPerm()), SysMenu::getPerm, queryParams.getPerm())
                .like(CharSequenceUtil.isNotBlank(queryParams.getPath()), SysMenu::getPath, queryParams.getPath())
                .eq(typeEnum != null, SysMenu::getType, typeEnum)
                .eq(queryParams.getVisible() != null, SysMenu::getVisible, queryParams.getVisible())
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
     * 新增/修改菜单（业务校验对齐 python _validate_menu_form：T-MM-015~031）
     */
    @Override
    public boolean saveMenu(MenuForm menuForm) {

        // 修改时检查菜单是否存在
        if (menuForm.getId() != null) {
            SysMenu existingMenu = this.getById(menuForm.getId());
            if (existingMenu == null) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在");
            }
            // 预置菜单保护：type 与 perm 分别是路由生成与接口鉴权的锚点，禁止修改
            if (Integer.valueOf(1).equals(existingMenu.getIsPreset())) {
                boolean typeChanged = menuForm.getType() != existingMenu.getType();
                boolean permChanged = !StringUtils.equals(menuForm.getPerm(), existingMenu.getPerm());
                if (typeChanged || permChanged) {
                    throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "系统预置菜单不可修改类型/权限标识");
                }
            }
        }

        MenuTypeEnum menuType = menuForm.getType();
        Long parentId = menuForm.getParentId() == null ? SystemConstants.ROOT_NODE_ID : menuForm.getParentId();
        String name = StrUtil.nullToEmpty(menuForm.getName());
        String path = StrUtil.nullToEmpty(menuForm.getPath());
        String perm = StrUtil.nullToEmpty(menuForm.getPerm());

        // T-MM-030：上级菜单不能是自己
        if (menuForm.getId() != null && menuForm.getId().equals(parentId)) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "上级菜单不能是自己");
        }

        if (parentId > 0) {
            SysMenu parent = this.getById(parentId);
            if (parent == null) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "父菜单不存在");
            }
            // T-MM-017/018：上级菜单不能是按钮/外链类型
            if (MenuTypeEnum.BUTTON.equals(parent.getType())) {
                throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "上级菜单不能是按钮类型");
            }
            if (MenuTypeEnum.EXTLINK.equals(parent.getType())) {
                throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "上级菜单不能是外链类型");
            }
            // T-MM-031：循环引用检测（不能将自己的子菜单设为父菜单）
            if (menuForm.getId() != null && isDescendant(menuForm.getId(), parentId)) {
                throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "不能设置自己的子菜单为父菜单");
            }
            // T-MM-022：层级限制（最多5级，根级为第1级）
            if (getMenuDepth(parentId) >= 5) {
                throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "菜单层级不能超过5级");
            }
        }

        // T-MM-015/027：同级菜单名称唯一（含软删行）
        if (this.baseMapper.countSameNameIncludeDeleted(parentId, name, menuForm.getId()) > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "菜单名称已存在");
        }

        // T-MM-016：权限标识全局唯一（含软删行）
        if (StringUtils.isNotBlank(perm)
                && this.baseMapper.countByPermIncludeDeleted(perm, menuForm.getId()) > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "权限标识已存在");
        }

        // T-MM-019/020：类型条件必填
        if ((menuType == MenuTypeEnum.MENU || menuType == MenuTypeEnum.CATALOG) && path.isEmpty()) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "路由地址不能为空");
        }
        if (menuType == MenuTypeEnum.EXTLINK && path.isEmpty()) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "外链地址不能为空");
        }
        if (menuType == MenuTypeEnum.BUTTON && perm.isEmpty()) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "权限标识不能为空");
        }

        if (menuType == MenuTypeEnum.CATALOG) {  // 如果是目录
            if (parentId == 0 && !path.startsWith("/")) {
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
                // 新增菜单默认分配给超级管理员（ROOT）与系统管理员（ADMIN），其他角色需手动分配
                for (String roleCode : List.of(SystemConstants.ROOT_ROLE_CODE, SystemConstants.ADMIN_ROLE_CODE)) {
                    SysRole role = roleMapper.selectOne(new LambdaQueryWrapper<SysRole>()
                            .eq(SysRole::getCode, roleCode));
                    if (role != null) {
                        roleMenuService.save(new SysRoleMenu(role.getId(), entity.getId()));
                    }
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
            // 按钮类型不显示在下拉选项中（对齐 python _build_menu_options）
            if (MenuTypeEnum.BUTTON.equals(menu.getType())) {
                continue;
            }
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
        List<RouteRead> menuList = this.baseMapper.listRoutes();
        Map<Long, List<RouteRead>> parentToChildrenMap = menuList.stream()
                .collect(Collectors.groupingBy(RouteRead::getParentId));
        List<RouteVO> routes = buildRoutes(SystemConstants.ROOT_NODE_ID, parentToChildrenMap);
        stringRedisTemplate.opsForValue().set(MENU_ROUTES_KEY, JSONUtil.toJsonStr(routes), MENU_CACHE_TTL);
        return routes;
    }

    /**
     * 递归生成菜单路由层级列表
     *
     * @param parentId           父级ID
     * @param parentToChildrenMap 父级ID → 子路由列表 Map（O(1)查找）
     */
    private List<RouteVO> buildRoutes(Long parentId, Map<Long, List<RouteRead>> parentToChildrenMap) {
        List<RouteVO> routeList = new ArrayList<>();
        List<RouteRead> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        for (RouteRead menu : children) {
            RouteVO routeVO = toRouteVo(menu);
            List<RouteVO> subRoutes = buildRoutes(menu.getId(), parentToChildrenMap);
            if (!subRoutes.isEmpty()) {
                routeVO.setChildren(subRoutes);
            }
            routeList.add(routeVO);
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
     * 根据RouteRead创建RouteVO
     */
    private RouteVO toRouteVo(RouteRead routeRead) {
        RouteVO routeVO = new RouteVO();
        String routeName = StringUtils.capitalize(CharSequenceUtil.toCamelCase(routeRead.getPath(), '-'));  // 路由 name 需要驼峰，首字母大写
        routeVO.setName(routeName); // 根据name路由跳转 this.$router.push({name:xxx})
        routeVO.setPath(routeRead.getPath()); // 根据path路由跳转 this.$router.push({path:xxx})
        routeVO.setRedirect(routeRead.getRedirect());
        routeVO.setComponent(routeRead.getComponent());

        RouteVO.Meta meta = new RouteVO.Meta();
        meta.setTitle(routeRead.getName());
        meta.setIcon(routeRead.getIcon());
        // 角色编码去重排序（对齐 python sorted(role_codes)，保证 meta.roles 顺序稳定可断言）
        meta.setRoles(routeRead.getRoles() == null ? Collections.emptyList()
                : routeRead.getRoles().stream().filter(Objects::nonNull).distinct().sorted().toList());
        meta.setHidden(StatusEnum.DISABLE.getValue().equals(routeRead.getVisible()));
        // 【菜单】是否开启页面缓存
        if (MenuTypeEnum.MENU.equals(routeRead.getType())
                && ObjectUtil.equals(routeRead.getKeepAlive(), 1)) {
            meta.setKeepAlive(true);
        }
        // 【目录】只有一个子路由是否始终显示
        if (MenuTypeEnum.CATALOG.equals(routeRead.getType())
                && ObjectUtil.equals(routeRead.getAlwaysShow(), 1)) {
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
        if (this.getById(menuId) == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在");
        }
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
        // 去重，避免重复 ID 导致存在性校验误判（T-MM-044）
        List<Long> distinctIds = ids.stream().distinct().toList();

        // 校验所有传入的菜单ID都存在
        long existCount = this.count(new LambdaQueryWrapper<SysMenu>().in(SysMenu::getId, distinctIds));
        if (existCount != distinctIds.size()) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在");
        }

        // 一次性查询所有待删除菜单ID（传入ID + 子孙），合并去重
        // 条件：id IN (ids) OR tree_path LIKE '%,id,%'（对每个 id 做 OR）
        LambdaQueryWrapper<SysMenu> wrapper = new LambdaQueryWrapper<SysMenu>()
                .in(SysMenu::getId, distinctIds);
        for (Long id : distinctIds) {
            wrapper.or().apply("CONCAT (',',tree_path,',') LIKE CONCAT('%,',{0},',%')", id);
        }
        List<Long> menuIds = this.list(wrapper).stream()
                .map(SysMenu::getId).distinct().toList();

        if (menuIds.isEmpty()) {
            return true;
        }

        // 预置菜单保护：级联命中范围内存在预置菜单时整批拒绝，不做部分删除
        long presetCount = this.count(new LambdaQueryWrapper<SysMenu>()
                .in(SysMenu::getId, menuIds)
                .eq(SysMenu::getIsPreset, 1));
        if (presetCount > 0) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "系统预置菜单不可删除");
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

    /**
     * 判断 targetId 是否为 ancestorId 的后代（其 tree_path 祖先链中包含 ancestorId）
     */
    private boolean isDescendant(Long ancestorId, Long targetId) {
        if (ancestorId.equals(targetId)) {
            return true;
        }
        SysMenu target = this.getById(targetId);
        if (target == null || StrUtil.isBlank(target.getTreePath())) {
            return false;
        }
        String ancestorStr = String.valueOf(ancestorId);
        return Arrays.stream(target.getTreePath().split(",")).anyMatch(ancestorStr::equals);
    }

    /**
     * 获取菜单层级（根级为第1级；tree_path 为祖先链，如根菜单 "0"、二级 "0,1"）
     */
    private int getMenuDepth(Long menuId) {
        if (SystemConstants.ROOT_NODE_ID.equals(menuId)) {
            return 0;
        }
        SysMenu menu = this.getById(menuId);
        if (menu == null || StrUtil.isBlank(menu.getTreePath())) {
            return 1;
        }
        return menu.getTreePath().split(",").length;
    }

}
