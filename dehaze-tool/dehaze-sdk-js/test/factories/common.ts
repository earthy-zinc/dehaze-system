import { faker } from "@faker-js/faker";
import { nanoid } from "nanoid";

/** 每次测试可重置种子，保证可复现 */
export const seedTestData = (seed: number = 20240101) => {
  faker.seed(seed);
  counter = 1;
};

let counter = 1;
const next = () => counter++;

/** 生成唯一名称 */
export const uniqueName = (prefix: string) => `${prefix}_${nanoid(6)}_${next()}`;

/** 生成唯一邮箱（使用时间戳确保跨测试运行唯一） */
export const uniqueEmail = (prefix = "test") => {
  const timestamp = Date.now().toString().slice(-8);
  const count = next();
  return `${prefix}_${timestamp}_${count}@example.com`;
};

/** 生成符合中国手机号格式的手机号（使用时间戳+计数器确保唯一） */
export const uniqueMobile = () => {
  // 中国手机号第二位只能是 3,4,5,6,7,8,9
  const secondDigits = ["3", "4", "5", "6", "7", "8", "9"];
  const secondDigit = faker.helpers.arrayElement(secondDigits);
  // 使用时间戳后6位 + 计数器，确保跨测试运行唯一
  const timestamp = Date.now().toString().slice(-6);
  const count = next().toString().padStart(3, "0");
  return `1${secondDigit}${timestamp}${count}`;
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
