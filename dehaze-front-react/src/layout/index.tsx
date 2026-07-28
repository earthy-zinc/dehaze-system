import { LayoutEnum } from "@/enums/LayoutEnum";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { usePermission } from "@/hooks/usePermission";
import { TopMenu } from "@/layout/components/MenuBar/TopMenu";
import { routesToRouteObjects } from "@/router";
import lazyLoad from "@/router/LazyLoad";
import { RootState } from "@/store";
import "./index.scss";
import { Layout, Spin } from "antd";
import { Content, Header } from "antd/es/layout/layout";
import Sider from "antd/es/layout/sider";
import React, { lazy, useMemo } from "react";
import { useSelector } from "react-redux";
import { Navigate, RouteObject, useRoutes } from "react-router-dom";
import { SideMenu } from "./components/MenuBar/SideMenu";
import { NavBar } from "./components/NavBar";

const SIDEBAR_WIDTH = 220;
const SIDEBAR_WIDTH_COLLAPSED = 64;

const BasicLayout: React.FC = () => {
  usePermission();
  const settingsStore = useSelector((state: RootState) => state.settings);
  const appStore = useSelector((state: RootState) => state.app);
  const permissionRoutes = useSelector(
    (state: RootState) => state.permission.routes
  );
  const collapsed = appStore.sidebarStatus === SidebarStatusEnum.COLLAPSED;
  const routesLoaded = permissionRoutes.length > 0;

  const routes = useMemo<RouteObject[]>(
    () => [
      { index: true, element: <Navigate to="home" replace /> },
      {
        path: "home",
        element: lazyLoad(lazy(() => import("@/pages/home"))),
      },
      {
        path: "message",
        element: lazyLoad(lazy(() => import("@/pages/message"))),
      },
      {
        path: "message/detail",
        element: lazyLoad(lazy(() => import("@/pages/message/detail"))),
      },
      {
        path: "message/settings",
        element: lazyLoad(lazy(() => import("@/pages/message/settings"))),
      },
      {
        path: "member/center",
        element: lazyLoad(lazy(() => import("@/pages/member/center"))),
      },
      {
        path: "member/growth-logs",
        element: lazyLoad(lazy(() => import("@/pages/member/growth-logs"))),
      },
      {
        path: "package/shop",
        element: lazyLoad(lazy(() => import("@/pages/package/shop"))),
      },
      {
        path: "order/my",
        element: lazyLoad(lazy(() => import("@/pages/order/my"))),
      },
      {
        path: "order/detail",
        element: lazyLoad(lazy(() => import("@/pages/order/detail"))),
      },
      {
        path: "feedback/my-ratings",
        element: lazyLoad(lazy(() => import("@/pages/feedback/my-ratings"))),
      },
      {
        path: "feedback/my",
        element: lazyLoad(lazy(() => import("@/pages/feedback/my"))),
      },
      {
        path: "feedback/detail",
        element: lazyLoad(lazy(() => import("@/pages/feedback/detail"))),
      },
      ...routesToRouteObjects(permissionRoutes),
      {
        path: "*",
        element: lazyLoad(lazy(() => import("@/pages/error/404"))),
      },
    ],
    [permissionRoutes]
  );

  const element = useRoutes(routes);

  return (
    <Layout className="main-container">
      {settingsStore.layout !== LayoutEnum.TOP && (
        <Sider
          collapsible
          collapsed={collapsed}
          width={SIDEBAR_WIDTH}
          collapsedWidth={SIDEBAR_WIDTH_COLLAPSED}
          className="side-bar"
          trigger={null}
        >
          <SideMenu />
        </Sider>
      )}
      <Layout className="layout-left">
        <Header className="header">
          {settingsStore.layout === LayoutEnum.TOP ? <TopMenu /> : <NavBar />}
        </Header>
        <Content>
          {routesLoaded ? (
            element
          ) : (
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                height: "100%",
              }}
            >
              <Spin size="large" />
            </div>
          )}
        </Content>
      </Layout>
    </Layout>
  );
};

export default BasicLayout;
