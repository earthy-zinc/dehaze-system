import CompareIcon from "@/assets/icons/compare.svg";
import DatasetIcon from "@/assets/icons/dataset.svg";
import HazeIcon from "@/assets/icons/haze.svg";
import ModelIcon from "@/assets/icons/model.svg";
import OverlapIcon from "@/assets/icons/overlap.svg";
import ParallelIcon from "@/assets/icons/parallel.svg";
import PresentationIcon from "@/assets/icons/presentation.svg";
import SegmentationIcon from "@/assets/icons/segmentation.svg";
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
