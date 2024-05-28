import { ConfigProvider, Watermark } from "antd";
import { SizeType } from "antd/es/config-provider/SizeContext";
import enUS from "antd/locale/en_US";
import zhCN from "antd/locale/zh_CN";
import { useMemo } from "react";
import { useSelector } from "react-redux";
import { RouterProvider } from "react-router-dom";

import router from "@/router";
import defaultSettings from "@/settings";
import { RootState } from "@/store";

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
    <ConfigProvider
      locale={locale}
      componentSize={appStore.size as SizeType}
      theme={{
        components: {
          Layout: {
            headerBg: "#fff",
            headerHeight: 50,
            headerPadding: 0,
            siderBg: "#fff",
          },
        },
      }}
    >
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
