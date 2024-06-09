import { RouteObject, createBrowserRouter, Link } from "react-router-dom";
import BasicLayout from "@/layout";
import { RouterInfo } from "@/typings/router";
import { ItemType } from "antd/es/menu/interface";
import {
  DatabaseOutlined,
  DeploymentUnitOutlined,
  HomeOutlined,
} from "@ant-design/icons";

const routerData: RouterInfo[] = [
  {
    path: "/",
    name: "/",
    component: <BasicLayout />,
    meta: {
      title: "首页",
      icon: <HomeOutlined />,
    },
    children: [
      {
        path: "/home",
        name: "Home",
        component: <div>首页</div>,
      },
    ],
  },
];

function buildRouteObjects(routes: RouterInfo[]): RouteObject[] {
  return routes.map((route) => {
    const routeObject: RouteObject = {
      path: route.path,
      element: route.component,
    };

    if (route.children) {
      routeObject.children = buildRouteObjects(route.children);
    }

    return routeObject;
  });
}

// 将RouterInfo转换为MenuItemProps的函数
function convertToMenuItem(route: RouterInfo): ItemType | null {
  if (!route.name || !route.path) {
    return null;
  }

  // 提取meta信息，如果没有则使用默认值
  const {
    title = route.name,
    icon = null,
    hidden = false,
    ...restMeta
  } = route.meta || {};

  // 根据RouteMeta中的hidden属性决定是否渲染该菜单项
  if (hidden) return null;

  // 转换icon为Ant Design的Icon组件，这里假设icon是React.ReactNode，如果是图标名称需转换为对应组件
  const menuIcon =
    typeof icon === "string" ? (
      <span className={`anticon anticon-${icon}`} />
    ) : (
      icon
    );
  // 如果当前路由有子路由，且子路由只有一个，则将子路由的path作为当前路由的path，并跳过子路由的渲染
  if (route.children && route.children.length === 1) {
    const childRoute = route.children[0];
    if (childRoute.path) {
      route.path = childRoute.path;
      route.children = undefined;
    }
  }

  return {
    key: route.path,
    label: title,
    icon: menuIcon,
    children: route.children?.map(convertToMenuItem).filter(Boolean), // 过滤掉null的子菜单项
  };
}
export const menuItems = routerData.map(convertToMenuItem).filter(Boolean);

const router = createBrowserRouter([
  {
    path: "/",
    element: <BasicLayout />,
    children: [
      {
        path: "/home",
        element: <div>首页</div>,
      },
    ],
  },
  {
    path: "*",
    element: <div>404</div>,
  },
]);

export default router;
