import AlgorithmIcon from "@/assets/icons/algorithm.svg";
import CompareIcon from "@/assets/icons/compare.svg";
import DatasetIcon from "@/assets/icons/dataset.svg";
import DictIcon from "@/assets/icons/dict.svg";
import EditIcon from "@/assets/icons/edit.svg";
import HazeIcon from "@/assets/icons/haze.svg";
import MenuIcon from "@/assets/icons/menu.svg";
import MessageIcon from "@/assets/icons/message.svg";
import ModelIcon from "@/assets/icons/model.svg";
import OverlapIcon from "@/assets/icons/overlap.svg";
import ParallelIcon from "@/assets/icons/parallel.svg";
import PresentationIcon from "@/assets/icons/presentation.svg";
import RoleIcon from "@/assets/icons/role.svg";
import SystemIcon from "@/assets/icons/system.svg";
import TaskIcon from "@/assets/icons/todolist.svg";
import TreeIcon from "@/assets/icons/tree.svg";
import UserIcon from "@/assets/icons/user.svg";
import BasicLayout from "@/layout";
import ErrorPage403 from "@/pages/error/403";
import ErrorPage404 from "@/pages/error/404";
import Login from "@/pages/login";
import Register from "@/pages/register";
import lazyLoad from "@/router/LazyLoad";
import {
  BarChartOutlined,
  BellOutlined,
  CrownOutlined,
  FileOutlined,
  FlagOutlined,
  GiftOutlined,
  MessageOutlined,
  PictureOutlined,
  ShopOutlined,
  UnorderedListOutlined,
  WalletOutlined,
} from "@ant-design/icons";
import React, { lazy } from "react";
import { createBrowserRouter, Navigate, RouteObject } from "react-router-dom";
import { RouteVO } from "dehaze-sdk-js";

const iconMap: Record<string, React.ReactNode> = {
  algorithm: <AlgorithmIcon />,
  announcement: <BellOutlined />,
  compare: <CompareIcon />,
  coupon: <GiftOutlined />,
  dataset: <DatasetIcon />,
  dict: <DictIcon />,
  edit: <EditIcon />,
  evaluation: <BarChartOutlined />,
  feedback: <MessageOutlined />,
  haze: <HazeIcon />,
  image: <PictureOutlined />,
  list: <UnorderedListOutlined />,
  member: <CrownOutlined />,
  menu: <MenuIcon />,
  message: <MessageIcon />,
  model: <ModelIcon />,
  order: <WalletOutlined />,
  overlap: <OverlapIcon />,
  package: <ShopOutlined />,
  parallel: <ParallelIcon />,
  presentation: <PresentationIcon />,
  role: <RoleIcon />,
  system: <SystemIcon />,
  template: <FileOutlined />,
  tree: <TreeIcon />,
  user: <UserIcon />,
  bell: <BellOutlined />,
  "el-icon-Flag": <FlagOutlined />,
  "el-icon-Files": <FileOutlined />,
};

export function getIcon(name?: string): React.ReactNode | undefined {
  if (!name) return undefined;
  return iconMap[name];
}

export function resolveFullPath(parentPath: string, childPath: string): string {
  if (!childPath) return parentPath;
  if (childPath.startsWith("/")) return childPath;
  if (!parentPath || parentPath === "/") return `/${childPath}`;
  return `${parentPath.replace(/\/$/, "")}/${childPath}`;
}

const pageModules = import.meta.glob("@/pages/**/index.tsx");

const lazyPages: Record<
  string,
  React.LazyExoticComponent<React.ComponentType<any>>
> = {};
for (const [key, loader] of Object.entries(pageModules)) {
  const path = key.replace(/^\/src\/pages\//, "").replace(/\.tsx$/, "");
  lazyPages[path] = lazy(
    loader as () => Promise<{ default: React.ComponentType<any> }>
  );
}

const ComingSoon = lazy(() => import("@/pages/coming-soon/index"));

function resolveComponent(
  componentPath: string
): React.LazyExoticComponent<React.ComponentType<any>> {
  return lazyPages[componentPath] || ComingSoon;
}

export function routesToRouteObjects(
  routes: RouteVO[],
  parentPath = ""
): RouteObject[] {
  return routes.map((route) => {
    const rawPath = route.path || "";
    const path = parentPath ? rawPath.replace(/^\//, "") : rawPath;
    const fullPath = resolveFullPath(parentPath, rawPath);
    const children: RouteObject[] = [];

    if (route.redirect) {
      children.push({
        index: true,
        element: <Navigate to={route.redirect} replace />,
      });
    }
    if (route.children?.length) {
      children.push(...routesToRouteObjects(route.children, fullPath));
    }

    const obj: RouteObject = { path };
    if (typeof route.component === "string" && route.component) {
      obj.element = lazyLoad(resolveComponent(route.component));
    }
    if (children.length) {
      obj.children = children;
    }
    return obj;
  });
}

export function routesToMenuItems(
  routes: RouteVO[],
  parentPath = ""
): {
  key: string;
  label: string;
  icon?: React.ReactNode;
  children?: unknown[];
}[] {
  return routes
    .filter((route) => !route.meta?.hidden)
    .map((route) => {
      const fullPath = resolveFullPath(parentPath, route.path || "");
      const visibleChildren = route.children?.filter(
        (child) => !child.meta?.hidden
      );
      const item: {
        key: string;
        label: string;
        icon?: React.ReactNode;
        children?: unknown[];
      } = {
        key: fullPath,
        label: route.meta?.title || "",
        icon: getIcon(route.meta?.icon),
      };
      if (visibleChildren?.length) {
        item.children = routesToMenuItems(visibleChildren, fullPath);
      }
      return item;
    });
}

const router = createBrowserRouter([
  {
    path: "/login",
    element: <Login />,
    errorElement: <ErrorPage404 />,
  },
  {
    path: "/register",
    element: <Register />,
    errorElement: <ErrorPage404 />,
  },
  {
    path: "/403",
    element: <ErrorPage403 />,
  },
  {
    path: "/*",
    element: <BasicLayout />,
    errorElement: <ErrorPage404 />,
  },
]);

export default router;
