import { PageQuery } from "@/types";

/** SKILL 目录内资源文件清单项（内容存对象存储，按需加载） */
export interface SkillFileVO {
  /** 相对 SKILL 根目录的文件路径（如 reference/REFERENCE.md） */
  path: string;
  /** 文件大小（字节） */
  fileSize: number;
  /** 文件类型（MIME/扩展名） */
  fileType?: string;
}

/** SKILL 上传时被跳过的资源文件（部分失败可见：文件名 + 原因） */
export interface SkillSkippedFile {
  /** 相对 SKILL 根目录的文件路径 */
  path: string;
  /** 跳过原因（越单文件上限 / 不在 SKILL 目录内等） */
  reason: string;
}

/** Skill 视图对象 */
export interface SkillVO {
  id: number;
  /** Skill 名称（唯一，遵循 Agent Skills 规范命名） */
  name: string;
  description?: string;
  /** 适用场景 */
  scene?: string;
  /** SKILL.md 指令正文（frontmatter 之外的内容；列表项不含，仅详情返回） */
  instruction?: string;
  /** SKILL.md frontmatter license（许可证） */
  license?: string;
  /** SKILL.md frontmatter compatibility（环境要求） */
  compatibility?: string;
  /** SKILL.md frontmatter metadata（任意键值，如版本/作者） */
  metadata?: Record<string, string>;
  /** SKILL.md frontmatter allowed-tools（预批准工具列表） */
  allowedTools?: string;
  /** SKILL 目录内资源文件清单（reference/script/assets，内容存对象存储） */
  files?: SkillFileVO[];
  /** 上传时被跳过的资源文件清单（仅上传/详情返回，文件名 + 原因） */
  skippedFiles?: SkillSkippedFile[];
  /** 状态：1-启用，0-禁用 */
  status: 0 | 1;
  /** 被 Agent 关联数 */
  agentCount?: number;
  /** 来源：builtin-内置播种，admin-管理员创建 */
  source?: "builtin" | "admin";
  /** 是否共享至市场（1-是，0-否） */
  marketShared?: number;
  createTime?: string;
  updateTime?: string;
}

/** 创建/更新 Skill 表单 */
export interface SkillForm {
  name: string;
  description?: string;
  scene?: string;
  /** Markdown 指令（内容校验：长度限制、危险操作拦截） */
  instruction: string;
}

/** Skill 分页查询参数 */
export interface SkillQuery extends PageQuery {
  keyword?: string;
  /** 状态筛选（1-启用，0-禁用；仅管理员全量列表生效） */
  status?: 0 | 1;
}

/** SKILL 市场目录项 */
export interface SkillMarketVO {
  /** Skill ID（市场内唯一） */
  skillId: number;
  name: string;
  description?: string;
  /** 适用场景 */
  scene?: string;
  /** 是否已启用 */
  enabled?: boolean;
  /** 已关联 Agent 数 */
  agentCount?: number;
}

/** Skill 试运行表单（独立调试会话，不入库不推送） */
export interface SkillTestForm {
  /** 测试输入数据（必填；字符串原样作为用户消息，其余序列化为 JSON） */
  inputData: unknown;
}

/** Skill 试运行结果：以 Skill 指令为系统提示词真实推理一次 */
export interface SkillTestResult {
  skillId: number;
  skillName: string;
  /** 本次试运行使用的指令全文 */
  instruction: string;
  /** 回显的测试输入 */
  input: unknown;
  /** 模型输出（final_response） */
  output: string;
  /** 本次推理用量 */
  usage: Record<string, unknown>;
}
