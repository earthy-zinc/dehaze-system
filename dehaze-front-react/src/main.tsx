import "@ant-design/v5-patch-for-react-19";
import App from "@/App";
import store, { persistor } from "@/store/index";
import React from "react";
import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import { PersistGate } from "redux-persist/integration/react";
import "@/styles/index.scss";
import "uno.css";
import "animate.css";
import configRequest from "./utils/request";

configRequest();

const root = ReactDOM.createRoot(document.getElementById("root")!);

root.render(
  <React.StrictMode>
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>
        <App />
      </PersistGate>
    </Provider>
  </React.StrictMode>
);
