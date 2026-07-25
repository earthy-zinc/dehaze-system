import CompareIcon from "@/assets/icons/compare.svg";
import DatasetIcon from "@/assets/icons/dataset.svg";
import HazeIcon from "@/assets/icons/haze.svg";
import ModelIcon from "@/assets/icons/model.svg";
import OverlapIcon from "@/assets/icons/overlap.svg";
import ParallelIcon from "@/assets/icons/parallel.svg";
import PresentationIcon from "@/assets/icons/presentation.svg";
import SegmentationIcon from "@/assets/icons/segmentation.svg";
import TaskIcon from "@/assets/icons/todolist.svg";
import BasicLayout from "@/layout";
import ErrorPage403 from "@/pages/error/403";
import ErrorPage404 from "@/pages/error/404";
import Login from "@/pages/login";
import Register from "@/pages/register";
import lazyLoad from "@/router/LazyLoad";
import {
  ApartmentOutlined,
  BookOutlined,
  HomeOutlined,
  MenuOutlined,
  SafetyOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import UserIcon from "@/assets/icons/user.svg";
import React, { lazy } from "react";
import { createBrowserRouter, Navigate } from "react-router-dom";

export const menuItems = [
  {
    key: "Home",
    label: "首页",
    icon: <HomeOutlined />,
    path: "/home",
  },
  {
    key: "Dataset",
    label: "数据集",
    icon: <DatasetIcon />,
    path: "/dataset",
  },
  {
    key: "Algorithm",
    label: "模型管理",
    icon: <ModelIcon />,
    path: "/algorithm",
  },
  {
    key: "Presentation",
    label: "算法展示",
    icon: <PresentationIcon />,
    path: "/presentation",
    children: [
      {
        key: "Dehaze",
        label: "图像去雾",
        icon: <HazeIcon />,
        path: "/presentation/dehaze",
      },
      {
        key: "Segmentation",
        label: "图像分割",
        icon: <SegmentationIcon />,
        path: "/presentation/segmentation",
      },
    ],
  },
  {
    key: "Compare",
    label: "算法比较",
    icon: <CompareIcon />,
    path: "/compare",
    children: [
      {
        key: "Overlap",
        label: "重叠对比",
        icon: <OverlapIcon />,
        path: "/compare/overlap",
      },
      {
        key: "Parallel",
        label: "并行对比",
        icon: <ParallelIcon />,
        path: "/compare/parallel",
      },
    ],
  },
  {
    key: "Task",
    label: "任务中心",
    icon: <TaskIcon />,
    path: "/task",
  },
  {
    key: "System",
    label: "系统管理",
    icon: <SettingOutlined />,
    path: "/system",
    children: [
      {
        key: "Dept",
        label: "部门管理",
        icon: <ApartmentOutlined />,
        path: "/system/dept",
      },
      {
        key: "Dict",
        label: "字典管理",
        icon: <BookOutlined />,
        path: "/system/dict",
      },
      {
        key: "Menu",
        label: "菜单管理",
        icon: <MenuOutlined />,
        path: "/system/menu",
      },
      {
        key: "Role",
        label: "角色管理",
        icon: <SafetyOutlined />,
        path: "/system/role",
      },
      {
        key: "User",
        label: "用户管理",
        icon: <UserIcon />,
        path: "/system/user",
      },
    ],
  },
];

const router = createBrowserRouter([
  {
    path: "/",
    element: <BasicLayout />,
    errorElement: <ErrorPage404 />,
    children: [
      {
        index: true,
        element: <Navigate to="home" replace />,
      },
      {
        path: "home",
        element: lazyLoad(lazy(() => import("@/pages/home"))),
      },
      {
        // 图像输入页（不在菜单中显示，通过导航跳转访问）
        path: "image-input",
        element: lazyLoad(lazy(() => import("@/pages/image-input"))),
      },
      {
        path: "dataset",
        children: [
          {
            index: true,
            element: lazyLoad(lazy(() => import("@/pages/dataset"))),
          },
          {
            path: ":id",
            element: lazyLoad(
              lazy(() => import("@/pages/dataset/DatasetDetail"))
            ),
          },
        ],
      },
      {
        path: "algorithm",
        children: [
          {
            index: true,
            element: lazyLoad(lazy(() => import("@/pages/algorithm"))),
          },
          {
            // 算法选择页（不在菜单中显示）
            path: "select",
            element: lazyLoad(lazy(() => import("@/pages/algorithm-select"))),
          },
        ],
      },
      {
        path: "presentation",
        children: [
          {
            index: true,
            element: <Navigate to="dehaze" replace />,
          },
          {
            path: "dehaze",
            element: lazyLoad(
              lazy(() => import("@/pages/presentation/dehaze"))
            ),
          },
          {
            path: "segmentation",
            element: lazyLoad(
              lazy(() => import("@/pages/presentation/segmentation"))
            ),
          },
        ],
      },
      {
        path: "compare",
        children: [
          {
            index: true,
            element: <Navigate to="overlap" replace />,
          },
          {
            path: "overlap",
            element: lazyLoad(lazy(() => import("@/pages/compare/overlap"))),
          },
          {
            path: "parallel",
            element: lazyLoad(lazy(() => import("@/pages/compare/parallel"))),
          },
        ],
      },
      {
        path: "task",
        element: lazyLoad(lazy(() => import("@/pages/task"))),
      },
      {
        path: "system",
        children: [
          {
            index: true,
            element: <Navigate to="dept" replace />,
          },
          {
            path: "dept",
            element: lazyLoad(lazy(() => import("@/pages/system/dept"))),
          },
          {
            path: "dict",
            element: lazyLoad(lazy(() => import("@/pages/system/dict"))),
          },
          {
            path: "menu",
            element: lazyLoad(lazy(() => import("@/pages/system/menu"))),
          },
          {
            path: "role",
            element: lazyLoad(lazy(() => import("@/pages/system/role"))),
          },
          {
            path: "user",
            element: lazyLoad(lazy(() => import("@/pages/system/user"))),
          },
        ],
      },
    ],
  },
  {
    path: "login",
    element: <Login />,
    errorElement: <ErrorPage404 />,
  },
  {
    path: "register",
    element: <Register />,
    errorElement: <ErrorPage404 />,
  },
  {
    path: "403",
    element: <ErrorPage403 />,
  },
  {
    path: "*",
    element: <ErrorPage404 />,
  },
]);

export default router;
