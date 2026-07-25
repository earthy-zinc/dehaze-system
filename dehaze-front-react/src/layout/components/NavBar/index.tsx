import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { DisPatchType, RootState } from "@/store";
import { toggleSidebar } from "@/store/modules/appSlice";
import { toggleSettingsVisible } from "@/store/modules/settingsSlice";
import { logout } from "@/store/modules/userSlice";
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import type { MenuProps } from "antd";
import { Breadcrumb, Dropdown, Modal } from "antd";
import "./index.scss";
import React from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import { Settings } from "./Settings";

export const NavBar: React.FC = () => {
  const appStore = useSelector((state: RootState) => state.app);
  const userStore = useSelector((state: RootState) => state.user);
  const dispatch = useDispatch<DisPatchType>();
  const navigate = useNavigate();
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

  // 注销登录：弹出确认框，确认后调用 logout 并跳转登录页
  const handleLogout = () => {
    Modal.confirm({
      title: "确认退出登录吗？",
      onOk: () => {
        dispatch(logout()).then(() => {
          navigate("/login", { replace: true });
        });
      },
    });
  };

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
          <div className="flex-center h100%">
            <img
              src={userStore.user.avatar}
              className="rounded-full ml-10px w24px h24px"
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
