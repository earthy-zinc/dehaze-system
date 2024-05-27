import { useSelector } from "react-redux";
import { useMemo } from "react";
import { RouterProvider } from "react-router-dom";

import { ConfigProvider, Watermark } from "antd";
import { SizeType } from "antd/es/config-provider/SizeContext";
import zhCN from "antd/locale/zh_CN";
import enUS from "antd/locale/en_US";

import defaultSettings from "@/settings";
import { RootState } from "@/store";
import router from "@/router";

function App() {
  const appStore = useSelector((state: RootState) => state.app);
  const settingsStore = useSelector((state: RootState) => state.settings);

  const locale = useMemo(() => {
    switch (appStore.language) {
      case "zh-CN":
        return zhCN;
      case "en-US":
        return enUS;
      default:
        return zhCN;
    }
  }, [appStore.language]);

  return (
    <ConfigProvider locale={locale} componentSize={appStore.size as SizeType}>
      <Watermark
        content={
          settingsStore.watermarkEnabled
            ? defaultSettings.watermarkContent
            : undefined
        }
      >
        <RouterProvider router={router} />
      </Watermark>
    </ConfigProvider>
  );
}

export default App;
