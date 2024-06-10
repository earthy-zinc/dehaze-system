import { LoadingOutlined } from "@ant-design/icons";
import { Spin } from "antd";
import React, { Suspense } from "react";

const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

const lazyLoad = (
  Component: React.LazyExoticComponent<any>
): React.ReactNode => (
  <Suspense
    fallback={
      <Spin
        indicator={antIcon}
        style={{
          position: "absolute",
          left: "50%",
          top: "50%",
        }}
      />
    }
  >
    <Component />
  </Suspense>
);

export default lazyLoad;
