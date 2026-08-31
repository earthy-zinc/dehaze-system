import { PageResult } from "@/types";
import request from "@/utils/request";
import type { AxiosHeaders } from "axios";
import type { SkillForm, SkillMarketVO, SkillQuery, SkillTestForm, SkillVO } from "./model";

/**
 * SKILL 管理 API。
 *
 * 列表接口普通用户仅返回启用项；管理操作需 ai:skill:manage。
 * Skill 执行链路（渐进式加载/按步骤执行）由 Agent 服务承载，本 API 负责管理侧。
 */
class AiSkillAPI {
  /** Skill 列表（管理员全量含停用；普通用户仅启用项） */
  static listSkills(query?: SkillQuery) {
    return request<PageResult<SkillVO[]>>({
      url: "/api/v1/ai/skills",
      method: "get",
      params: query,
    });
  }

  /** 创建 Skill（遵循 Agent Skills 规范的最小目录：SKILL.md） */
  static createSkill(data: SkillForm) {
    return request<SkillVO>({
      url: "/api/v1/ai/skills",
      method: "post",
      data,
    });
  }

  /**
   * 上传 SKILL 压缩包（zip，遵循 Agent Skills 规范：SKILL.md + reference/script/assets）。
   * 后端自动解析 SKILL.md frontmatter 并校验（name 命名/目录一致/description），
   * 文件内容存对象存储，元数据与文件清单入库。
   */
  static uploadSkill(file: File | Blob, filename?: string) {
    const fd = new FormData();
    const name = filename || (file instanceof File ? file.name : "skill.zip");
    fd.append("file", file, name);
    return request<SkillVO>({
      url: "/api/v1/ai/skills/upload",
      method: "post",
      data: fd,
      // 置空默认的 application/json，由 axios 按 multipart/form-data 处理（Node 端补 boundary）
      headers: { "Content-Type": undefined } as unknown as AxiosHeaders,
    });
  }

  /** Skill 详情 */
  static getSkill(id: number) {
    return request<SkillVO>({
      url: `/api/v1/ai/skills/${id}`,
      method: "get",
    });
  }

  /** 读取 SKILL 资源文件内容（须命中该 Skill 的文件清单，返回 Blob） */
  static getSkillFile(id: number, path: string) {
    return request<Blob>({
      url: `/api/v1/ai/skills/${id}/file`,
      method: "get",
      params: { path },
      responseType: "blob",
    });
  }

  /** 更新 Skill（新会话生效，进行中会话沿用旧版本） */
  static updateSkill(id: number, data: Partial<SkillForm>) {
    return request<SkillVO>({
      url: `/api/v1/ai/skills/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除 Skill（软删除；校验是否被 Agent 关联，有则提示先解绑） */
  static deleteSkill(id: number) {
    return request({
      url: `/api/v1/ai/skills/${id}`,
      method: "delete",
    });
  }

  /** 启停 Skill（status: 1 启用 / 0 禁用） */
  static switchSkillStatus(id: number, status: 0 | 1) {
    return request<SkillVO>({
      url: `/api/v1/ai/skills/${id}/status`,
      method: "patch",
      data: { status },
    });
  }

  /** 试运行 Skill（测试数据不入库不推送） */
  static testSkill(id: number, data: SkillTestForm) {
    return request<Record<string, unknown>>({
      url: `/api/v1/ai/skills/${id}/test`,
      method: "post",
      data,
    });
  }

  // ==================== 市场 ====================

  /** SKILL 市场目录（预设/共享 Skill） */
  static getMarket() {
    return request<SkillMarketVO[]>({
      url: "/api/v1/ai/skills/market",
      method: "get",
    });
  }

  /** 将自建 Skill 共享至市场（需先启用） */
  static shareToMarket(id: number) {
    return request<SkillVO>({
      url: "/api/v1/ai/skills/market",
      method: "post",
      data: { skillId: id },
    });
  }
}

export default AiSkillAPI;
