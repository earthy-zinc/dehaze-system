import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import defaultSettings from "@/settings";
import { RootState } from "@/store";
import React from "react";
import "./index.scss";
import { useSelector } from "react-redux";
import { Link } from "react-router-dom";

export const Logo: React.FC = () => {
  const appStore = useSelector((state: RootState) => state.app);
  const collapsed = appStore.sidebarStatus === SidebarStatusEnum.COLLAPSED;
  const title = defaultSettings.title;
  return (
    <div className="logo-container">
      <Link className="logo" to={"/"}>
        <img src="/logo/logo.png" alt="logo" />
      </Link>

      {!collapsed && <h1>{title}</h1>}
    </div>
  );
};
