import React, { useEffect } from "react";
import configRequest from "@/utils/request";
import { useAuthStore } from "@/stores/global";
import "./app.less";

// 应用最外层起始处配置请求拦截器：模块加载即执行，早于任何组件渲染与网络请求，
// 与桌面端 dehaze-front-react 入口（main.tsx）保持一致
configRequest();

interface AppProps {
  children: React.ReactNode;
}

// 内部组件，处理认证初始化
const AppContent: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const initAuth = useAuthStore((s) => s.initAuth);

  useEffect(() => {
    // 应用启动时初始化认证状态，仅执行一次（initAuth 为 store 稳定引用）
    initAuth();
  }, [initAuth]);

  return <>{children}</>;
};

const App: React.FC<AppProps> = (props) => {
  // 底部导航采用原生 tabBar（app.config.ts 配置），由各端框架渲染，无需全局自绘组件
  return <AppContent>{props.children}</AppContent>;
};

export default App;
