import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { RootState } from "@/store";
import { toggleSidebar } from "@/store/modules/appSlice";
import { toggleSettingsVisible } from "@/store/modules/settingsSlice";
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import type { MenuProps } from "antd";
import { Breadcrumb, Dropdown } from "antd";
import "./index.scss";
import React from "react";
import { useDispatch, useSelector } from "react-redux";
import { Settings } from "./Settings";

export const NavBar: React.FC = () => {
  const appStore = useSelector((state: RootState) => state.app);
  const userStore = useSelector((state: RootState) => state.user);
  const dispatch = useDispatch();
  const items: MenuProps["items"] = [
    {
      key: "1",
      label: "注销登录",
      onClick: () => handleLogout(),
    },
  ];
  const MenuStatus =
    appStore.sidebarStatus === SidebarStatusEnum.OPENED ? (
      <MenuUnfoldOutlined />
    ) : (
      <MenuFoldOutlined />
    );

  const handleLogout = () => {};

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
        <Dropdown className="settings-item" menu={{ items }}>
          <div className="flex-center h100% p10px">
            <img
              src={userStore.user.avatar}
              className="rounded-full mr-10px w24px h24px"
              alt=""
            />
            <span style={{ minWidth: "60px" }}>{userStore.user.username}</span>
          </div>
        </Dropdown>
        <button className="menu-status-icon" onClick={handleSettingClick}>
          <SettingOutlined />
        </button>
      </div>
      <Settings />
    </>
  );
};
