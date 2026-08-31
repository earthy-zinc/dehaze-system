<!-- 管理端 SKILL 管理页：SKILL 市场（一键启用）+ SKILL 管理（CRUD/启停/试运行） -->
<script lang="ts" setup>
import { SkillVO } from "dehaze-sdk-js";
import { Plus, Refresh } from "@element-plus/icons-vue";
import { useAdminSkillStore } from "@/store/modules/adminSkill";
import MarketSkillList from "./components/MarketSkillList.vue";
import SkillFormDialog from "./components/SkillFormDialog.vue";
import SkillTable from "./components/SkillTable.vue";
import SkillTestPanel from "./components/SkillTestPanel.vue";

// 页面名需与动态路由名（由组件路径推导为 AiSkills）一致，否则 keep-alive 缓存失效
defineOptions({ name: "AiSkills" });

const skillStore = useAdminSkillStore();
const activeTab = ref<"market" | "manage">("market");

async function handleSaved(skill: SkillVO) {
  // 创建即试用：保存后引导试运行，验证指令效果再交付使用
  try {
    await ElMessageBox.confirm(
      "是否立即试运行该 Skill 验证指令效果？",
      "保存成功",
      {
        type: "success",
        confirmButtonText: "试运行",
        cancelButtonText: "稍后",
      }
    );
  } catch {
    return;
  }
  skillStore.openTestPanel(skill);
}

onMounted(() => {
  skillStore.refreshAll();
});
</script>

<template>
  <div class="app-container">
    <el-card shadow="never" class="!border-none">
      <template #header>
        <div class="flex items-center justify-between">
          <span class="font-bold">SKILL 管理</span>
          <div>
            <el-button
              v-hasPerm="['ai:skill:manage']"
              type="success"
              @click="skillStore.openFormDialog(null)"
            >
              <el-icon><Plus /></el-icon>
              新建 Skill
            </el-button>
            <el-button @click="skillStore.refreshAll()">
              <el-icon><Refresh /></el-icon>
              刷新
            </el-button>
          </div>
        </div>
      </template>

      <el-tabs v-model="activeTab">
        <el-tab-pane label="SKILL 市场" name="market">
          <MarketSkillList />
        </el-tab-pane>
        <el-tab-pane label="SKILL 管理" name="manage">
          <SkillTable
            @edit="skillStore.openFormDialog"
            @test="skillStore.openTestPanel"
          />
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <SkillFormDialog @saved="handleSaved" />
    <SkillTestPanel />
  </div>
</template>
