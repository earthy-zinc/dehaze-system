import React from "react";
import "./index.scss";
import { Link } from "react-router-dom";
import { useSelector } from "react-redux";
import { RootState } from "@/store";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";

export const Logo: React.FC = () => {
  const appStore = useSelector((state: RootState) => state.app);
  const collapsed = appStore.sidebarStatus === SidebarStatusEnum.COLLAPSED;

  return (
    <div className="logo-container">
      <Link className="logo" to={"/"}>
        <img
          src="https://gw.alipayobjects.com/zos/antfincdn/PmY%24TNNDBI/logo.svg"
          alt="logo"
        />
      </Link>

      {!collapsed && <h1>Ant Design Pro</h1>}
    </div>
  );
};
