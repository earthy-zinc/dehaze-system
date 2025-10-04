import { LayoutEnum } from "@/enums/LayoutEnum";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
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

const BasicLayout: React.FC = (props: any) => {
  const settingsStore = useSelector((state: RootState) => state.settings);
  const appStore = useSelector((state: RootState) => state.app);
  const collapsed = appStore.sidebarStatus === SidebarStatusEnum.COLLAPSED;
  const sidebarWidthCollapsed = Number(
    window
      .getComputedStyle(document.documentElement)
      .getPropertyValue("--sidebar-width-collapsed")
      .replace("px", "")
  );
  return (
    <Layout className="main-container">
      {settingsStore.layout !== LayoutEnum.TOP && (
        <Sider
          collapsible
          collapsed={collapsed}
          collapsedWidth={sidebarWidthCollapsed}
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
