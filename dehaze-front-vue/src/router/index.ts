import { createRouter, createWebHashHistory, RouteRecordRaw } from "vue-router";

export const Layout = () => import("@/layout/index.vue");

// 静态路由
export const constantRoutes: RouteRecordRaw[] = [
  {
    path: "/redirect",
    component: Layout,
    meta: { hidden: true },
    children: [
      {
        path: "/redirect/:path(.*)",
        component: () => import("@/views/redirect/index.vue"),
      },
    ],
  },

  {
    path: "/login",
    component: () => import("@/views/login/index.vue"),
    meta: { hidden: true },
  },
  {
    path: "/register",
    component: () => import("@/views/register/index.vue"),
    meta: { hidden: true },
  },

  {
    path: "/",
    name: "/",
    component: Layout,
    redirect: "/home",
    children: [
      {
        path: "home",
        component: () => import("@/views/home/index.vue"),
        name: "Home", // 用于 keep-alive，必须与SFC自动推导或者显示声明的组件name一致
        meta: {
          title: "首页",
          icon: "homepage",
          hidden: true,
          affix: true,
        },
      },
      {
        path: "dashboard",
        component: () => import("@/views/dashboard/index.vue"),
        name: "Dashboard", // 用于 keep-alive, 必须与SFC自动推导或者显示声明的组件name一致
        // https://cn.vuejs.org/guide/built-ins/keep-alive.html#include-exclude
        meta: {
          title: "dashboard",
          icon: "homepage",
          hidden: true,
          affix: true,
          keepAlive: true,
          alwaysShow: false,
        },
      },
      {
        path: "401",
        component: () => import("@/views/error-page/401.vue"),
        meta: { hidden: true },
      },
      {
        path: "404",
        component: () => import("@/views/error-page/404.vue"),
        meta: { hidden: true },
      },
      {
        path: "notify/message",
        component: () => import("@/views/notify/message/index.vue"),
        name: "NotifyMessage",
        meta: {
          title: "消息中心",
          icon: "message",
          hidden: true,
          keepAlive: true,
        },
      },
      {
        path: "notify/message/detail",
        component: () => import("@/views/notify/message/detail.vue"),
        name: "NotifyMessageDetail",
        meta: { title: "消息详情", hidden: true },
      },
      {
        path: "notify/settings",
        component: () => import("@/views/notify/settings/index.vue"),
        name: "NotifySettings",
        meta: {
          title: "通知设置",
          icon: "setting",
          hidden: true,
          keepAlive: true,
        },
      },
      {
        path: "member/center",
        component: () => import("@/views/member/center/index.vue"),
        name: "MemberCenter",
        meta: {
          title: "会员中心",
          icon: "member",
          hidden: true,
          keepAlive: true,
        },
      },
      {
        path: "member/growth-logs",
        component: () => import("@/views/member/growth-logs/index.vue"),
        name: "MemberGrowthLogs",
        meta: { title: "成长值明细", hidden: true, keepAlive: true },
      },
      {
        path: "package/shop",
        component: () => import("@/views/package/shop/index.vue"),
        name: "PackageShop",
        meta: {
          title: "套餐购买",
          icon: "package",
          hidden: true,
          keepAlive: true,
        },
      },
      {
        path: "order/my",
        component: () => import("@/views/order/my/index.vue"),
        name: "OrderMy",
        meta: {
          title: "我的订单",
          icon: "order",
          hidden: true,
          keepAlive: true,
        },
      },
      {
        path: "order/detail",
        component: () => import("@/views/order/detail/index.vue"),
        name: "OrderDetail",
        meta: { title: "订单详情", hidden: true },
      },
      {
        path: "feedback/my-ratings",
        component: () => import("@/views/feedback/my-ratings/index.vue"),
        name: "FeedbackMyRatings",
        meta: { title: "我的评价", hidden: true, keepAlive: true },
      },
      {
        path: "feedback/my",
        component: () => import("@/views/feedback/my/index.vue"),
        name: "FeedbackMy",
        meta: { title: "我的反馈", hidden: true, keepAlive: true },
      },
      {
        path: "feedback/detail",
        component: () => import("@/views/feedback/detail/index.vue"),
        name: "FeedbackDetail",
        meta: { title: "反馈详情", hidden: true },
      },
    ],
  },
];

/**
 * 创建路由
 */
const router = createRouter({
  history: createWebHashHistory(),
  routes: constantRoutes,
  // 刷新时，滚动条位置还原
  scrollBehavior: () => ({ left: 0, top: 0 }),
});

/**
 * 重置路由
 */
export function resetRouter() {
  router.replace({ path: "/login" });
}

export default router;
