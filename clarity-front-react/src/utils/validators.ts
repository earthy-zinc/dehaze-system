import type { Rule } from "antd/es/form";

export function required(message?: string): Rule {
  return { required: true, message: message || "该项为必填" };
}

export function emailRule(message?: string): Rule {
  return { type: "email", message: message || "请输入有效的邮箱地址" };
}

export function urlRule(message?: string): Rule {
  return { type: "url", message: message || "请输入有效的URL" };
}

export function phoneRule(message?: string): Rule {
  return {
    pattern: /^1[3-9]\d{9}$/,
    message: message || "请输入有效的手机号",
  };
}

export function minLen(n: number, message?: string): Rule {
  return { min: n, message: message || `最少${n}个字符` };
}

export function maxLen(n: number, message?: string): Rule {
  return { max: n, message: message || `最多${n}个字符` };
}
