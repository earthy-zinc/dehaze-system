import router from "@/router";
import defaultSettings from "@/settings";
import { RootState } from "@/store";
import { ConfigProvider, theme, Watermark } from "antd";
import { SizeType } from "antd/es/config-provider/SizeContext";
import enUS from "antd/locale/en_US";
import zhCN from "antd/locale/zh_CN";
import { useMemo } from "react";
import { useSelector } from "react-redux";
import { RouterProvider } from "react-router-dom";
import { ThemeEnum } from "./enums/ThemeEnum";
import useSystemTheme from "./hooks/useSystemTheme";

function App() {
  const appStore = useSelector((state: RootState) => state.app);
  const settingsStore = useSelector((state: RootState) => state.settings);
  const systemTheme = useSystemTheme();

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

  const algorithm = useMemo(() => {
    const customTheme =
      settingsStore.theme === ThemeEnum.AUTO
        ? systemTheme
        : settingsStore.theme;
    const customAlgorithm =
      customTheme === ThemeEnum.LIGHT
        ? [theme.defaultAlgorithm]
        : [theme.darkAlgorithm];
    if (appStore.size === "small") {
      customAlgorithm.push(theme.compactAlgorithm);
    }
    return customAlgorithm;
  }, [appStore.size, settingsStore.theme, systemTheme]);

  return (
    <ConfigProvider
      locale={locale}
      componentSize={appStore.size as SizeType}
      theme={{
        algorithm,
        cssVar: true,
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
        style={{ width: "100%", height: "100%", overflow: "auto" }}
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
