import "@ant-design/v5-patch-for-react-19";
import App from "@/App";
import store, { persistor } from "@/store/index";
import React from "react";
import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import { PersistGate } from "redux-persist/integration/react";
import { ErrorBoundary, Logger, ConsoleTransport, RemoteTransport } from "dehaze-sdk-js";
import "@/styles/index.scss";
import "uno.css";
import "animate.css";
import configRequest from "./utils/request";

// 前端日志监控：注册全局错误捕获 + 离线上报。SDK 不感知环境，
// 由应用端按构建产物组装 transports（开发仅 Console，生产追加 Remote）
Logger.install({
  app: "react",
  appVersion: __APP_INFO__.pkg.version,
  transports: import.meta.env.PROD
    ? [new ConsoleTransport(), new RemoteTransport()]
    : [new ConsoleTransport()],
  react: React,
});

configRequest();

const root = ReactDOM.createRoot(document.getElementById("root")!);

root.render(
  <React.StrictMode>
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </PersistGate>
    </Provider>
  </React.StrictMode>
);
