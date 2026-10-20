import router from "@/router";
import defaultSettings from "@/settings";
import { RootState } from "@/store";
import TitleBar from "@/components/TitleBar";
import {
  App as AntdApp,
  ConfigProvider,
  message,
  theme,
  Watermark,
} from "antd";
import { SizeType } from "antd/es/config-provider/SizeContext";
import enUS from "antd/locale/en_US";
import zhCN from "antd/locale/zh_CN";
import React, { useEffect, useMemo } from "react";
import { useSelector } from "react-redux";
import { RouterProvider } from "react-router-dom";
import { ThemeEnum } from "./enums/ThemeEnum";
import useSystemTheme from "./hooks/useSystemTheme";

/** 品牌主色 */
const BRAND_PRIMARY = "#3B82F6";

function App() {
  const [messageApi, contextHolder] = message.useMessage();
  const appStore = useSelector((state: RootState) => state.app);
  const settingsStore = useSelector((state: RootState) => state.settings);
  const systemTheme = useSystemTheme();

  const isElectron = !!window.electronAPI;

  useEffect(() => {
    document.documentElement.style.setProperty(
      "--titlebar-h",
      isElectron ? "var(--titlebar-height)" : "0px"
    );
  }, [isElectron]);

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
    <>
      {contextHolder}
      {isElectron && <TitleBar />}
      <ConfigProvider
        locale={locale}
        componentSize={appStore.size as SizeType}
        theme={{
          algorithm,
          cssVar: true,
          token: {
            // 品牌主色
            colorPrimary: BRAND_PRIMARY,
            // 链接色
            colorLink: BRAND_PRIMARY,
            // 圆角：卡片 16px，输入 8px，按钮 10px
            borderRadius: 8,
            borderRadiusLG: 16,
            borderRadiusSM: 6,
            // 字体栈
            fontFamily:
              "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif",
            // 阴影
            boxShadow:
              "0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)",
            boxShadowSecondary:
              "0 4px 12px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.06)",
          },
          components: {
            Layout: {
              headerBg: "#fff",
              headerHeight: 56,
              headerPadding: "0 24px",
              siderBg: "#fff",
              bodyBg: "#f5f7fa",
            },
            Menu: {
              itemHeight: 44,
              iconSize: 18,
              itemMarginInline: 8,
              itemBorderRadius: 8,
              activeBarHeight: 0,
              activeBarBorderWidth: 0,
              subMenuItemBg: "transparent",
            },
            Card: {
              borderRadiusLG: 16,
              paddingLG: 24,
              boxShadowTertiary:
                "0 2px 8px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04)",
            },
            Button: {
              borderRadius: 10,
              controlHeight: 36,
              controlHeightLG: 44,
              controlHeightSM: 30,
              primaryShadow: "0 2px 6px rgba(59, 130, 246, 0.3)",
            },
            Input: {
              borderRadius: 8,
              controlHeight: 36,
            },
            Table: {
              borderRadius: 12,
              headerBg: "#fafbfc",
              headerColor: "#1f2937",
              cellPaddingBlock: 14,
              cellPaddingInline: 16,
            },
            Modal: {
              borderRadiusLG: 16,
            },
          },
        }}
      >
        <AntdApp>
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
        </AntdApp>
      </ConfigProvider>
    </>
  );
}

export default App;
