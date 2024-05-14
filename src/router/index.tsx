import React from "react";
import { useRoutes, useNavigate, Navigate } from "react-router-dom";
import routes, { routeType } from "./routes";

export default function Routes() {
  // return useRoutes(renderRoutes(routes));
}

function renderRoutes(routes: routeType[]) {
  return routes.map((route) => {
    const res = { ...route };
    if (!route.path) return;
    const { path, children, ...rest } = route;
    return res;
  });
}
