import { faker } from "@faker-js/faker";
import { customAlphabet, nanoid } from "nanoid";

let counter = 1;
const next = () => counter++;

// 仅字母数字的 nanoid 字母表：后端注册用户名等字段有 ^[a-zA-Z0-9_]+$ 类校验，
// nanoid 默认字母表含 "-"，混入会导致唯一名被 A0400 拒绝
const alnumNano = customAlphabet("useandom26T198340PX75pxJACKVERYMINDBUSHWOLFGQZbfghjklqvwyzrict");

/** 生成唯一名称 */
export const uniqueName = (prefix: string) => `${prefix}_${alnumNano(6)}_${next()}`;

/**
 * 生成合法注册用户名（3-32 位，仅字母、数字、下划线，与后端校验一致）。
 * 前缀过长时截断，保证随机段与计数器完整保留。
 */
export const uniqueUsername = (prefix: string) => {
  const safePrefix = prefix.replace(/[^a-zA-Z0-9_]/g, "").slice(0, 20);
  return `${safePrefix}_${alnumNano(6)}_${next()}`.slice(0, 32);
};

/** 生成唯一邮箱（使用时间戳确保跨测试运行唯一） */
export const uniqueEmail = (prefix = "test") => {
  const timestamp = Date.now().toString().slice(-8);
  const count = next();
  return `${prefix}_${timestamp}_${count}@example.com`;
};

/** 生成符合中国手机号格式的手机号（使用时间戳+计数器确保跨测试运行唯一）
 * 号段必须匹配后端 UserForm 的 @Pattern 校验：
 * ^$|^1(3\d|4[5-9]|5[0-35-9]|6[2567]|7[0-8]|8\d|9[0-35-9])\d{8}$
 */
export const uniqueMobile = () => {
  // 合法号段前缀（3位），覆盖后端 Pattern 允许的全部号段
  const prefixes = [
    "130",
    "131",
    "132",
    "133",
    "134",
    "135",
    "136",
    "137",
    "138",
    "139",
    "145",
    "146",
    "147",
    "148",
    "149",
    "150",
    "151",
    "152",
    "153",
    "155",
    "156",
    "157",
    "158",
    "159",
    "162",
    "165",
    "166",
    "167",
    "170",
    "171",
    "172",
    "173",
    "174",
    "175",
    "176",
    "177",
    "178",
    "180",
    "181",
    "182",
    "183",
    "184",
    "185",
    "186",
    "187",
    "188",
    "189",
    "190",
    "191",
    "192",
    "193",
    "195",
    "196",
    "197",
    "198",
    "199",
  ];
  const prefix = faker.helpers.arrayElement(prefixes);
  // 后8位：时间戳后6位 + 计数器2位，保证跨测试运行唯一
  const timestamp = Date.now().toString().slice(-6);
  const count = next().toString().padStart(2, "0");
  return `${prefix}${timestamp}${count}`;
};

/** 生成通用编码 */
export const uniqueCode = (prefix = "CODE") => `${prefix}_${nanoid(8)}`;

/** 默认分页查询生成器，确保类型安全 */
export const pageQuery = <T extends { pageNum?: number; pageSize?: number }>(
  overrides?: Partial<T>
): T => ({
  pageNum: 1,
  pageSize: 10,
  ...(overrides as T),
});
