import { createBrowserRouter } from "react-router-dom";
import BasicLayout from "@/layout";

const router = createBrowserRouter([
  {
    path: "/",
    element: <BasicLayout />,
    children: [],
  },
]);

export default router;
