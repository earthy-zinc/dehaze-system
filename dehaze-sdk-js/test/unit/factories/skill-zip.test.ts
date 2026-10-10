import { buildSkillZip } from "#/factories/ai-skill";
import { expect, test } from "vitest";

/**
 * SKILL zip 工厂的结构基线。
 *
 * 覆盖它的 ai-skill 集成测试依赖真实服务，本地无法运行；而 zip 结构错（如 EOCD 条目数写错）
 * 在 Node 侧不会报错、只会在服务端解析时暴露，故此处直接守住字节结构。
 */
test("buildSkillZip 产出结构自洽的 STORE 压缩 zip", () => {
  const zip = buildSkillZip();
  const eocd = zip.subarray(zip.length - 22);

  // EOCD 签名与条目数须与实际写入的三个条目一致（原实现用 entries.length 恒写 0）
  expect(eocd.readUInt32LE(0)).toBe(0x06054b50);
  expect(eocd.readUInt16LE(8)).toBe(3);
  expect(eocd.readUInt16LE(10)).toBe(3);

  // 中央目录长度须与其起始偏移互补，否则解析器定位不到条目
  expect(eocd.readUInt32LE(12)).toBe(zip.length - 22 - eocd.readUInt32LE(16));
});
