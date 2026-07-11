import { Button, Result } from "antd";
import React from "react";
import { useNavigate } from "react-router-dom";

// 403 无权限页面
export default function ErrorPage403() {
  const navigate = useNavigate();

  return (
    <Result
      status="403"
      title="403"
      subTitle="抱歉，您没有权限访问该页面。"
      extra={
        <Button type="primary" onClick={() => navigate("/", { replace: true })}>
          返回首页
        </Button>
      }
    />
  );
}
