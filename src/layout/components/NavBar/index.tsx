import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { RootState } from "@/store";
import { toggleSidebar } from "@/store/modules/appSlice";
import { toggleSettingsVisible } from "@/store/modules/settingsSlice";
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import { Breadcrumb } from "antd";
import "./index.scss";
import React from "react";
import { useDispatch, useSelector } from "react-redux";
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
    dispatch(
      toggleSidebar(
        appStore.sidebarStatus === SidebarStatusEnum.OPENED
          ? SidebarStatusEnum.COLLAPSED
          : SidebarStatusEnum.OPENED
      )
    );
  };
  const handleSettingClick = () => {
    dispatch(toggleSettingsVisible());
  };

  return (
    <>
      <div className="navbar-left">
        <button className="menu-status-icon" onClick={handleMenuStatusChange}>
          {MenuStatus}
        </button>
        <Breadcrumb items={[{ title: "首页" }]} />
      </div>
      <div className="navbar-right">
        <button className="menu-status-icon" onClick={handleSettingClick}>
          <SettingOutlined />
        </button>
      </div>
      <Settings />
    </>
  );
};
