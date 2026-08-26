import { PageQuery } from "@/types";

/** Skill 视图对象 */
export interface SkillVO {
  id: number;
  /** Skill 名称（唯一） */
  name: string;
  description?: string;
  /** 适用场景 */
  scene?: string;
  /** Markdown 指令内容 */
  instruction: string;
  /** 可选脚本/模板内容 */
  scriptContent?: string;
  /** 可选模板 ID */
  templateId?: number;
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
