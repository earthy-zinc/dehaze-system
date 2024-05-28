import { Layout } from "antd";
import { Header, Content } from "antd/es/layout/layout";
import Sider from "antd/es/layout/Sider";
import { useSelector } from "react-redux";

import { LayoutEnum } from "@/enums/LayoutEnum";
import { RootState } from "@/store";
import "./index.scss";
import { SideMenu } from "./components/MenuBar/SideMenu";
import { NavBar } from "./components/NavBar";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";

const BasicLayout: React.FC = (props: any) => {
  const settingsStore = useSelector((state: RootState) => state.settings);
  const appStore = useSelector((state: RootState) => state.app);
  const collapsed = appStore.sidebarStatus === SidebarStatusEnum.COLLAPSED;

  return (
    <Layout className="main-container">
      {settingsStore.layout === LayoutEnum.TOP && (
        <Sider collapsible collapsed={collapsed} className="side-bar">
          <SideMenu />
        </Sider>
      )}
      <Layout className="layout-left">
        <Header className="header">
          <NavBar />
        </Header>
        <Content>内容区</Content>
      </Layout>
    </Layout>
  );
};

export default BasicLayout;
