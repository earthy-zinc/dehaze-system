import {
  AiSkillAPI,
  SkillForm,
  SkillMarketVO,
  SkillQuery,
  SkillVO,
} from "dehaze-sdk-js";

// 管理端 SKILL Store：市场目录、Skill 列表/表单/试运行
export const useAdminSkillStore = defineStore("adminSkill", () => {
  const marketSkills = ref<SkillMarketVO[]>([]);
  const marketLoading = ref(false);

  const skills = ref<SkillVO[]>([]);
  const total = ref(0);
  const loading = ref(false);
  const query = reactive<SkillQuery>({ pageNum: 1, pageSize: 10 });

  const skillForm = reactive<{ visible: boolean; skill: SkillVO | null }>({
    visible: false,
    skill: null,
  });
  const submitting = ref(false);

  const testDialog = reactive<{ visible: boolean; skill: SkillVO | null }>({
    visible: false,
    skill: null,
  });
  const testResult = ref<Record<string, unknown> | null>(null);
  const testLoading = ref(false);

  /** 市场启用/共享会同时影响市场目录与管理列表，两端一并刷新 */
  async function refreshAll() {
    await Promise.all([fetchMarketSkills(), fetchSkills()]);
  }

  async function fetchMarketSkills() {
    marketLoading.value = true;
    try {
      marketSkills.value = await AiSkillAPI.getMarket();
    } finally {
      marketLoading.value = false;
    }
  }

  async function fetchSkills() {
    loading.value = true;
    try {
      const page = await AiSkillAPI.listSkills(query);
      skills.value = page.list ?? [];
      total.value = page.total ?? 0;
    } finally {
      loading.value = false;
    }
  }

  /** 市场一键启用：市场项即已共享的 Skill，启用后用户端会话即可按需加载 */
  async function installMarketSkill(skillId: number) {
    await AiSkillAPI.switchSkillStatus(skillId, 1);
    ElMessage.success("Skill 已启用");
    await refreshAll();
  }

  function openFormDialog(skill: SkillVO | null) {
    skillForm.skill = skill;
    skillForm.visible = true;
  }

  function openTestPanel(skill: SkillVO) {
    testDialog.skill = skill;
    testDialog.visible = true;
    testResult.value = null;
  }

  async function saveSkill(form: SkillForm) {
    submitting.value = true;
    try {
      const saved = skillForm.skill
        ? await AiSkillAPI.updateSkill(skillForm.skill.id, form)
        : await AiSkillAPI.createSkill(form);
      await refreshAll();
      return saved;
    } finally {
      submitting.value = false;
    }
  }

  /** zip 压缩包上传创建 SKILL（Agent Skills 规范，multipart） */
  async function uploadSkill(file: File) {
    submitting.value = true;
    try {
      const saved = await AiSkillAPI.uploadSkill(file);
      await refreshAll();
      return saved;
    } finally {
      submitting.value = false;
    }
  }

  /** 拉取 Skill 详情（含 frontmatter 字段与文件清单） */
  async function fetchSkillDetail(id: number): Promise<SkillVO> {
    return AiSkillAPI.getSkill(id);
  }

  /** 读取 SKILL 资源文件内容（Blob，供预览/下载） */
  async function getSkillFile(id: number, path: string): Promise<Blob> {
    return AiSkillAPI.getSkillFile(id, path);
  }

  async function switchSkillStatus(skill: SkillVO, status: 0 | 1) {
    await AiSkillAPI.switchSkillStatus(skill.id, status);
    await refreshAll();
  }

  async function deleteSkill(skill: SkillVO) {
    await AiSkillAPI.deleteSkill(skill.id);
    await refreshAll();
  }

  async function testSkill(skillId: number, inputData: unknown) {
    testLoading.value = true;
    try {
      testResult.value = await AiSkillAPI.testSkill(skillId, { inputData });
    } finally {
      testLoading.value = false;
    }
  }

  async function shareSkillToMarket(skill: SkillVO) {
    await AiSkillAPI.shareToMarket(skill.id);
    ElMessage.success("已共享至市场");
    await refreshAll();
  }

  return {
    marketSkills,
    marketLoading,
    skills,
    total,
    loading,
    query,
    skillForm,
    submitting,
    testDialog,
    testResult,
    testLoading,
    refreshAll,
    fetchMarketSkills,
    fetchSkills,
    installMarketSkill,
    openFormDialog,
    openTestPanel,
    saveSkill,
    uploadSkill,
    fetchSkillDetail,
    getSkillFile,
    switchSkillStatus,
    deleteSkill,
    testSkill,
    shareSkillToMarket,
  };
});
