import { LayoutEnum } from "@/enums/LayoutEnum";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { usePermission } from "@/hooks/usePermission";
import { TopMenu } from "@/layout/components/MenuBar/TopMenu";
import { RootState } from "@/store";
import "./index.scss";
import { Layout } from "antd";
import { Content, Header } from "antd/es/layout/layout";
import Sider from "antd/es/layout/Sider";
import React from "react";
import { useSelector } from "react-redux";
import { Outlet } from "react-router-dom";
import { SideMenu } from "./components/MenuBar/SideMenu";
import { NavBar } from "./components/NavBar";

/** 侧边栏宽度（参考 UI/UX 设计规范 §3.2 侧边菜单） */
const SIDEBAR_WIDTH = 220;
const SIDEBAR_WIDTH_COLLAPSED = 64;

const BasicLayout: React.FC = (props: any) => {
  // 激活路由守卫：未登录跳转 /login，登录后自动拉取用户信息与动态路由
  usePermission();
  const settingsStore = useSelector((state: RootState) => state.settings);
  const appStore = useSelector((state: RootState) => state.app);
  const collapsed = appStore.sidebarStatus === SidebarStatusEnum.COLLAPSED;
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
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
};

export default BasicLayout;
