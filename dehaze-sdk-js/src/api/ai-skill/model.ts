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

/** Skill 视图对象 */
export interface SkillVO {
  id: number;
  /** Skill 名称（唯一，遵循 Agent Skills 规范命名） */
  name: string;
  description?: string;
  /** 适用场景 */
  scene?: string;
  /** SKILL.md 指令正文（frontmatter 之外的内容） */
  instruction: string;
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
  /** 状态：1-启用，0-禁用 */
  status: 0 | 1;
  /** 被 Agent 关联数 */
  agentCount?: number;
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
  scriptContent?: string;
  templateId?: number;
  /** 状态：1-启用，0-禁用 */
  status?: 0 | 1;
}

/** Skill 分页查询参数 */
export interface SkillQuery extends PageQuery {
  keyword?: string;
  /** 状态筛选（1-启用，0-禁用） */
  status?: 0 | 1;
}

/** SKILL 市场目录项 */
export interface SkillMarketVO {
  /** Skill ID（市场内唯一） */
  skillId: number;
  name: string;
  description?: string;
  scene?: string;
  /** 是否已启用 */
  enabled?: boolean;
  /** 已关联 Agent 数 */
  agentCount?: number;
}

/** Skill 试运行表单（测试数据不入库不推送） */
export interface SkillTestForm {
  /** 测试输入数据 */
  inputData: unknown;
}
