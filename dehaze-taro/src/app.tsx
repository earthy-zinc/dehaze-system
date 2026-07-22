import React, { useEffect } from "react";
import configRequest from "@/utils/request";
import { GlobalProvider } from "@/stores/global";
import { useAuth } from "@/hooks/useAuth";
import "./app.less";

interface AppProps {
  children: React.ReactNode;
}

// 内部组件，处理认证初始化
const AppContent: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { initAuth } = useAuth();

  useEffect(() => {
    // 应用启动时初始化认证状态
    initAuth();
  }, [initAuth]);

  return <>{children}</>;
};

const App: React.FC<AppProps> = (props) => {
  // 配置请求拦截器（仅执行一次）
  useEffect(() => {
    configRequest();
  }, []);

  return (
    <GlobalProvider>
      <AppContent>{props.children}</AppContent>
    </GlobalProvider>
  );
};

export default App;
