import BasicLayout from "@/layout";
import ErrorPage404 from "@/pages/error/404";
import Login from "@/pages/login";
import lazyLoad from "@/router/LazyLoad";
import { HomeOutlined } from "@ant-design/icons";
import { lazy } from "react";
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
    icon: <HomeOutlined />,
    path: "/dataset",
  },
  {
    key: "Algorithm",
    label: "模型管理",
    icon: <HomeOutlined />,
    path: "/algorithm",
  },
  {
    key: "Presentation",
    label: "算法展示",
    icon: <HomeOutlined />,
    path: "/presentation",
    children: [
      {
        key: "Dehaze",
        label: "图像去雾",
        icon: <HomeOutlined />,
        path: "/presentation/dehaze",
      },
      {
        key: "Segmentation",
        label: "图像分割",
        icon: <HomeOutlined />,
        path: "/presentation/segmentation",
      },
    ],
  },
  {
    key: "Compare",
    label: "算法比较",
    icon: <HomeOutlined />,
    path: "/compare",
    children: [
      {
        key: "Overlap",
        label: "重叠对比",
        icon: <HomeOutlined />,
        path: "/compare/overlap",
      },
      {
        key: "Parallel",
        label: "并行对比",
        icon: <HomeOutlined />,
        path: "/compare/parallel",
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
    ],
  },
  {
    path: "login",
    element: <Login />,
    errorElement: <ErrorPage404 />,
  },
  {
    path: "*",
    element: <ErrorPage404 />,
  },
]);

export default router;
