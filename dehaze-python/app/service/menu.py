from app.extensions import mysql
from app.models import SysMenu, SysRoleMenu
from typing import Optional, List, Dict, Any, Set
from sqlalchemy import and_, or_, func


class MenuService:
    """菜单服务类，处理菜单相关的业务逻辑"""

    @staticmethod
    def list_menus(keywords: str = None) -> List[Dict[str, Any]]:
        """
        获取菜单列表（树形结构）
        
        Args:
            keywords (str, optional): 搜索关键字（菜单名称）
            
        Returns:
            List[Dict[str, Any]]: 菜单列表
        """
        # 查询所有菜单，按排序字段升序排列
        query = SysMenu.query.order_by(SysMenu.sort)
        
        if keywords:
            query = query.filter(SysMenu.name.like(f'%{keywords}%'))
            
        menus = query.all()
        
        # 构建菜单树
        return MenuService._build_menu_tree(0, menus)

    @staticmethod
    def _build_menu_tree(parent_id: int, menus: List[SysMenu]) -> List[Dict[str, Any]]:
        """
        递归构建菜单树
        
        Args:
            parent_id (int): 父级菜单ID
            menus (List[SysMenu]): 菜单列表
            
        Returns:
            List[Dict[str, Any]]: 树形菜单列表
        """
        tree = []
        for menu in menus:
            if menu.parent_id == parent_id:
                menu_dict = {
                    'id': menu.id,
                    'parentId': menu.parent_id,
                    'name': menu.name,
                    'type': menu.type,
                    'path': menu.path,
                    'component': menu.component,
                    'perm': menu.perm,
                    'visible': menu.visible,
                    'sort': menu.sort,
                    'icon': menu.icon,
                    'redirect': menu.redirect,
                    'alwaysShow': menu.always_show,
                    'keepAlive': menu.keep_alive,
                    'createTime': menu.create_time.isoformat() if menu.create_time else None
                }
                
                # 递归查找子菜单
                children = MenuService._build_menu_tree(menu.id, menus)
                if children:
                    menu_dict['children'] = children
                    
                tree.append(menu_dict)
                
        return tree

    @staticmethod
    def list_menu_options() -> List[Dict[str, Any]]:
        """
        获取菜单下拉选项列表
        
        Returns:
            List[Dict[str, Any]]: 菜单下拉选项列表
        """
        menus = SysMenu.query.order_by(SysMenu.sort).all()
        return MenuService._build_menu_options(0, menus)

    @staticmethod
    def _build_menu_options(parent_id: int, menus: List[SysMenu]) -> List[Dict[str, Any]]:
        """
        递归构建菜单下拉选项
        
        Args:
            parent_id (int): 父级菜单ID
            menus (List[SysMenu]): 菜单列表
            
        Returns:
            List[Dict[str, Any]]: 菜单下拉选项列表
        """
        options = []
        for menu in menus:
            if menu.parent_id == parent_id:
                option = {
                    'value': menu.id,
                    'label': menu.name
                }
                
                # 递归查找子菜单选项
                children = MenuService._build_menu_options(menu.id, menus)
                if children:
                    option['children'] = children
                    
                options.append(option)
                
        return options

    @staticmethod
    def save_menu(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存菜单（新增/修改）
        
        Args:
            data (Dict[str, Any]): 菜单数据
            
        Returns:
            Dict[str, Any]: 保存结果
        """
        menu_id = data.get('id')
        
        # 检查菜单是否存在（更新时）
        if menu_id:
            menu = SysMenu.query.get(menu_id)
            if not menu:
                return {'error': '菜单不存在'}
        else:
            menu = SysMenu()
            
        # 设置菜单属性
        menu.parent_id = data.get('parentId', 0)
        menu.name = data.get('name', '')
        menu.type = data.get('type', 1)
        menu.path = data.get('path', '')
        menu.component = data.get('component')
        menu.perm = data.get('perm')
        menu.visible = data.get('visible', 1)
        menu.sort = data.get('sort', 0)
        menu.icon = data.get('icon', '')
        menu.redirect = data.get('redirect')
        menu.always_show = data.get('alwaysShow')
        menu.keep_alive = data.get('keepAlive')
        
        # 生成树路径
        tree_path = MenuService._generate_menu_tree_path(menu.parent_id)
        menu.tree_path = tree_path
        
        # 设置默认值
        if menu.type == 2:  # 目录类型
            if menu.parent_id == 0 and not menu.path.startswith('/'):
                menu.path = '/' + menu.path
            menu.component = 'Layout'
        elif menu.type == 3:  # 外链类型
            menu.component = None
            
        try:
            if menu_id:
                mysql.session.merge(menu)
            else:
                mysql.session.add(menu)
                
            mysql.session.commit()
            return {'data': {'id': menu.id}}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'保存菜单失败: {str(e)}'}

    @staticmethod
    def _generate_menu_tree_path(parent_id: int) -> str:
        """
        生成菜单树路径
        
        Args:
            parent_id (int): 父级菜单ID
            
        Returns:
            str: 树路径，格式如 "0,1,2"
        """
        if parent_id == 0:
            return '0'
        else:
            parent_menu = SysMenu.query.get(parent_id)
            if parent_menu and parent_menu.tree_path:
                return f'{parent_menu.tree_path},{parent_menu.id}'
            else:
                return str(parent_id)

    @staticmethod
    def list_routes() -> List[Dict[str, Any]]:
        """
        获取路由列表
        
        Returns:
            List[Dict[str, Any]]: 路由列表
        """
        # 查询所有有效的菜单（类型为目录或菜单，且可见）
        menus = SysMenu.query.filter(
            SysMenu.type.in_([1, 2]),  # 1:菜单, 2:目录
            SysMenu.visible == 1
        ).order_by(SysMenu.sort).all()
        
        # 构建路由树
        return MenuService._build_routes(0, menus)

    @staticmethod
    def _build_routes(parent_id: int, menus: List[SysMenu]) -> List[Dict[str, Any]]:
        """
        递归构建路由列表
        
        Args:
            parent_id (int): 父级菜单ID
            menus (List[SysMenu]): 菜单列表
            
        Returns:
            List[Dict[str, Any]]: 路由列表
        """
        routes = []
        for menu in menus:
            if menu.parent_id == parent_id:
                # 构建路由对象
                route = MenuService._to_route_vo(menu)
                
                # 递归查找子路由
                children = MenuService._build_routes(menu.id, menus)
                if children:
                    route['children'] = children
                    
                routes.append(route)
                
        return routes

    @staticmethod
    def _to_route_vo(menu: SysMenu) -> Dict[str, Any]:
        """
        将菜单转换为路由对象
        
        Args:
            menu (SysMenu): 菜单对象
            
        Returns:
            Dict[str, Any]: 路由对象
        """
        # 路由name需要驼峰命名，首字母大写
        route_name = ''.join(word.capitalize() for word in menu.path.replace('-', '_').split('_') if word)
        
        route = {
            'name': route_name,
            'path': menu.path,
            'redirect': menu.redirect,
            'component': menu.component
        }
        
        # 构建meta信息
        meta = {
            'title': menu.name,
            'icon': menu.icon,
            'hidden': menu.visible == 0
        }
        
        # 【菜单】是否开启页面缓存
        if menu.type == 1 and menu.keep_alive == 1:  # 1:菜单
            meta['keepAlive'] = True
            
        # 【目录】只有一个子路由是否始终显示
        if menu.type == 2 and menu.always_show == 1:  # 2:目录
            meta['alwaysShow'] = True
            
        route['meta'] = meta
        return route

    @staticmethod
    def update_menu_visible(menu_id: int, visible: int) -> Dict[str, Any]:
        """
        更新菜单显示状态
        
        Args:
            menu_id (int): 菜单ID
            visible (int): 显示状态（1:显示; 0:隐藏）
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        if visible not in [0, 1]:
            return {'error': '显示状态只能为0或1'}
            
        menu = SysMenu.query.get(menu_id)
        if not menu:
            return {'error': '菜单不存在'}
            
        menu.visible = visible
        try:
            mysql.session.commit()
            return {'data': '更新成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'更新失败: {str(e)}'}

    @staticmethod
    def list_role_perms(roles: Set[str]) -> Set[str]:
        """
        获取角色权限集合
        
        Args:
            roles (Set[str]): 角色编码集合
            
        Returns:
            Set[str]: 权限集合
        """
        # 在Python版本中，我们通过角色获取权限的方式略有不同
        # 这里简化处理，实际项目中应该关联角色和菜单权限表
        role_menus = mysql.session.query(SysRoleMenu).join(
            SysMenu, SysRoleMenu.menu_id == SysMenu.id
        ).filter(
            SysMenu.perm.isnot(None),
            SysMenu.perm != ''
        ).all()
        
        perms = set()
        for role_menu in role_menus:
            menu = SysMenu.query.get(role_menu.menu_id)
            if menu and menu.perm:
                perms.add(menu.perm)
                
        return perms

    @staticmethod
    def get_menu_form(menu_id: int) -> Optional[Dict[str, Any]]:
        """
        获取菜单表单数据
        
        Args:
            menu_id (int): 菜单ID
            
        Returns:
            Optional[Dict[str, Any]]: 菜单表单数据
        """
        menu = SysMenu.query.get(menu_id)
        if not menu:
            return None
            
        return {
            'id': menu.id,
            'parentId': menu.parent_id,
            'name': menu.name,
            'type': menu.type,
            'path': menu.path,
            'component': menu.component,
            'perm': menu.perm,
            'visible': menu.visible,
            'sort': menu.sort,
            'icon': menu.icon,
            'redirect': menu.redirect,
            'alwaysShow': menu.always_show,
            'keepAlive': menu.keep_alive
        }

    @staticmethod
    def delete_menu(menu_id: int) -> Dict[str, Any]:
        """
        删除菜单
        
        Args:
            menu_id (int): 菜单ID
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        menu = SysMenu.query.get(menu_id)
        if not menu:
            return {'error': '菜单不存在'}
            
        try:
            # 删除菜单及其子菜单
            SysMenu.query.filter(
                or_(
                    SysMenu.id == menu_id,
                    func.concat(',', SysMenu.tree_path, ',').like(f'%,{menu_id},%')
                )
            ).delete(synchronize_session=False)
            
            mysql.session.commit()
            return {'data': '删除成功'}
        except Exception as e:
            mysql.session.rollback()
            return {'error': f'删除失败: {str(e)}'}