import { pageQuery } from "./common";
import { uniqueName } from "./common";
import type { SkillForm, SkillQuery, SkillTestForm } from "../../src/api/ai-skill/model";

/** Skill 创建表单工厂（名称前缀 test_skill_ 便于清理） */
export const createSkillForm = (overrides?: Partial<SkillForm>): SkillForm => ({
  name: uniqueName("test_skill"),
  description: "SKILL 管理契约测试",
  scene: "通用",
  instruction: "# 测试 Skill 指令\n按步骤执行测试流程",
  ...overrides,
});

/** Skill 分页查询参数工厂 */
export const createSkillQuery = (overrides?: Partial<SkillQuery>) =>
  pageQuery<SkillQuery>({ ...overrides });

/** Skill 试运行表单工厂 */
export const createSkillTestForm = (overrides?: Partial<SkillTestForm>): SkillTestForm => ({
  inputData: { text: "测试输入" },
  ...overrides,
});

/* ==================== SKILL zip 上传（Agent Skills 规范） ==================== */

let _crcTable: Int32Array | undefined;

/** CRC32（STORE 压缩 zip 校验，测试构造用） */
function crc32(data: Buffer): number {
  if (!_crcTable) {
    _crcTable = new Int32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) {
        c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      }
      _crcTable[n] = c;
    }
  }
  let crc = -1;
  for (let i = 0; i < data.length; i++) {
    crc = (crc >>> 8) ^ _crcTable[(crc ^ data[i]) & 0xff];
  }
  return (crc ^ -1) >>> 0;
}

/** 构造遵循 Agent Skills 规范的 SKILL 目录 zip（STORE 无压缩，内容可覆盖） */
export const buildSkillZip = (files?: Record<string, string>): Buffer => {
  const entries = files ?? {
    "pdf-extract/SKILL.md":
      "---\n" +
      "name: pdf-extract\n" +
      "description: 提取 PDF 文本与表格，处理 PDF 文档时使用\n" +
      "license: Apache-2.0\n" +
      "---\n" +
      "# PDF 提取步骤\n" +
      "1. 读取文件\n" +
      "2. 提取文本\n",
    "pdf-extract/script/extract.py": "print('hi')",
    "pdf-extract/reference/REFERENCE.md": "# 参考文档\n详细说明",
  };
  const locals: Buffer[] = [];
  const centrals: Buffer[] = [];
  let offset = 0;
  for (const [name, content] of Object.entries(entries)) {
    const nameBuf = Buffer.from(name, "utf-8");
    const dataBuf = Buffer.from(content, "utf-8");
    const crc = crc32(dataBuf);
    const lfh = Buffer.alloc(30);
    lfh.writeUInt32LE(0x04034b50, 0);
    lfh.writeUInt16LE(20, 4);
    lfh.writeUInt16LE(0, 8); // method: STORE
    lfh.writeUInt32LE(crc, 14);
    lfh.writeUInt32LE(dataBuf.length, 18);
    lfh.writeUInt32LE(dataBuf.length, 22);
    lfh.writeUInt16LE(nameBuf.length, 26);
    const local = Buffer.concat([lfh, nameBuf, dataBuf]);
    locals.push(local);
    const cdh = Buffer.alloc(46);
    cdh.writeUInt32LE(0x02014b50, 0);
    cdh.writeUInt16LE(20, 4);
    cdh.writeUInt16LE(20, 6);
    cdh.writeUInt32LE(crc, 16);
    cdh.writeUInt32LE(dataBuf.length, 20);
    cdh.writeUInt32LE(dataBuf.length, 24);
    cdh.writeUInt16LE(nameBuf.length, 28);
    cdh.writeUInt32LE(offset, 42);
    centrals.push(Buffer.concat([cdh, nameBuf]));
    offset += local.length;
  }
  const cdBuf = Buffer.concat(centrals);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(cdBuf.length, 12);
  eocd.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cdBuf, eocd]);
};

/** 生成符合 Agent Skills 命名规范（小写字母数字连字符）的唯一 Skill 名 */
export const uniqueSkillName = (): string =>
  `skillzip${Date.now().toString(36)}${Math.floor(Math.random() * 0xffff).toString(36)}`;

/** SKILL zip 上传文件工厂（File，测试环境为 Node；name 唯一避免残留冲突） */
export const createSkillZipFile = (): File => {
  const name = uniqueSkillName();
  const files: Record<string, string> = {
    [`${name}/SKILL.md`]:
      "---\n" +
      `name: ${name}\n` +
      "description: 提取 PDF 文本与表格，处理 PDF 文档时使用\n" +
      "license: Apache-2.0\n" +
      "metadata:\n" +
      '  version: "1.0"\n' +
      "---\n" +
      "# PDF 提取步骤\n" +
      "1. 读取文件\n",
    [`${name}/script/extract.py`]: "print('hi')",
    [`${name}/reference/REFERENCE.md`]: "# 参考文档\n详细说明",
  };
  return new File([buildSkillZip(files)], `${name}.zip`, { type: "application/zip" });
};
