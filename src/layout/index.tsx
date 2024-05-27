import Layout, { Header, Footer, Content } from "antd/lib/layout/layout";
import classNames from "classnames";
import styles from "./index.module.scss";
import { RootState } from "@/store";
import { useSelector } from "react-redux";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";
import { DeviceEnum } from "@/enums/DeviceEnum";

const BasicLayout: React.FC = (props: any) => {
  const appStore = useSelector((state: RootState) => state.app);
  const settingsStore = useSelector((state: RootState) => state.settings);

  const classObj = classNames({
    [styles.hideSidebar]: appStore.sidebarStatus !== SidebarStatusEnum.OPENED,
    [styles.openSidebar]: appStore.sidebarStatus === SidebarStatusEnum.OPENED,
    [styles.mobile]: appStore.device === DeviceEnum.MOBILE,
    [styles.layoutLeft]: settingsStore.layout === "left",
    [styles.layoutTop]: settingsStore.layout === "top",
    [styles.layoutMix]: settingsStore.layout === "mix",
  });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
      }}
      className={classObj}
    >
      lala
    </div>
  );
};

export default BasicLayout;
