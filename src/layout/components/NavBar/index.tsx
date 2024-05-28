import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import { useDispatch, useSelector } from "react-redux";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { RootState } from "@/store";
import { Breadcrumb } from "antd";
import "./index.scss";
import { Settings } from "./Settings";

export const NavBar: React.FC = () => {
  const appStore = useSelector((state: RootState) => state.app);
  const dispatch = useDispatch();
  const MenuStatus =
    appStore.sidebarStatus === SidebarStatusEnum.OPENED ? (
      <MenuUnfoldOutlined />
    ) : (
      <MenuFoldOutlined />
    );

  const handleMenuStatusChange = () => {
    dispatch({
      type: "app/toggleSidebar",
      payload:
        appStore.sidebarStatus === SidebarStatusEnum.OPENED
          ? SidebarStatusEnum.COLLAPSED
          : SidebarStatusEnum.OPENED,
    });
  };
  const handleSettingClick = () => {
    dispatch({
      type: "settings/toggleSettingsVisiable",
    });
  };
  return (
    <>
      <div className="navbar-left">
        <div className="menu-status-icon" onClick={handleMenuStatusChange}>
          {MenuStatus}
        </div>
        <Breadcrumb items={[{ title: "首页" }]} />
      </div>
      <div className="navbar-right">
        <div className="menu-status-icon" onClick={handleSettingClick}>
          <SettingOutlined />
        </div>
      </div>
      <Settings />
    </>
  );
};
