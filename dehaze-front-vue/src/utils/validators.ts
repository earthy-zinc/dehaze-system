import type { FormItemRule } from "element-plus";

type ValidateCallback = (error?: string) => void;

/**
 * 生成 Element Plus Form 校验规则
 * @param rule 校验规则配置
 * @param value 校验值
 * @param callback 回调函数
 */
function createValidator(
  rule: Partial<FormItemRule>,
  value: any,
  callback: ValidateCallback
) {
  const msg = typeof rule.message === "string" ? rule.message : undefined;

  if (rule.validator) {
    (rule.validator as any)(rule, value, callback);
    return;
  }

  if (rule.required && !value && value !== 0 && value !== false) {
    callback(msg || "此项为必填项");
    return;
  }

  if (value == null || value === "") {
    callback();
    return;
  }

  const strValue = String(value);

  if (rule.min != null && strValue.length < rule.min) {
    callback(msg || `最少${rule.min}个字符`);
    return;
  }

  if (rule.max != null && strValue.length > rule.max) {
    callback(msg || `最多${rule.max}个字符`);
    return;
  }

  if (rule.pattern) {
    const regex =
      rule.pattern instanceof RegExp ? rule.pattern : new RegExp(rule.pattern);
    if (!regex.test(strValue)) {
      callback(msg || "格式不正确");
      return;
    }
  }

  callback();
}

/**
 * 必填校验
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isRequired(message?: string) {
  return { required: true, message: message || "此项为必填项" };
}

/**
 * 邮箱格式校验
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isValidEmail(message?: string) {
  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return { pattern, message: message || "请输入有效的邮箱地址" };
}

/**
 * 中国大陆手机号校验
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isValidPhone(message?: string) {
  const pattern = /^1[3-9]\d{9}$/;
  return { pattern, message: message || "请输入有效的手机号码" };
}

/**
 * URL 格式校验
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isValidUrl(message?: string) {
  const pattern =
    /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/;
  return { pattern, message: message || "请输入有效的 URL 地址" };
}

/**
 * 最小长度校验
 * @param min 最小长度
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isMinLength(min: number, message?: string) {
  return { min, message: message || `最少${min}个字符` };
}

/**
 * 最大长度校验
 * @param max 最大长度
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isMaxLength(max: number, message?: string) {
  return { max, message: message || `最多${max}个字符` };
}

/**
 * 纯数字校验
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isNumber(message?: string) {
  const pattern = /^\d+(\.\d+)?$/;
  return { pattern, message: message || "请输入纯数字" };
}

/**
 * 整数校验
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isInteger(message?: string) {
  const pattern = /^\d+$/;
  return { pattern, message: message || "请输入整数" };
}

/**
 * 正数校验
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isPositiveNumber(message?: string) {
  const pattern = /^[1-9]\d*(\.\d+)?$/;
  return { pattern, message: message || "请输入正数" };
}

/**
 * 合法文件名校验（不含 \ / : * ? " < > |）
 * @param message 自定义错误消息
 * @returns Element Plus 校验规则
 */
export function isValidFileName(message?: string) {
  const pattern = /^[^\\/:*?"<>|\r\n]+$/;
  return { pattern, message: message || "文件名包含非法字符" };
}
