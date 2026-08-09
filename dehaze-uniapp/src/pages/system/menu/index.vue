<template>
  <PageLayout level="L2" title="菜单管理">
    <view class="page-body">
      <view v-for="menu in menuTree" :key="menu.id" class="tree-node">
        <view class="tree-item" @click="toggleExpand(menu)">
          <text class="tree-label"
            >{{ menu.name
            }}<text class="tree-type">[{{ typeLabel(menu.type) }}]</text></text
          >
          <view class="tree-actions" @click.stop>
            <SvgIcon v-if="canEdit" name="edit-pen" @click="editMenu(menu)" />
            <SvgIcon
              v-if="canDelete"
              name="trash"
              color="#ef4444"
              @click="delMenu(menu.id)"
            />
          </view>
          <SvgIcon
            v-if="menu.children?.length"
            :name="
              expandedIds.includes(menu.id!) ? 'arrow-down' : 'arrow-right'
            "
            size="16"
          />
        </view>
        <view
          v-if="expandedIds.includes(menu.id!) && menu.children"
          class="tree-children"
        >
          <view
            v-for="child in menu.children"
            :key="child.id"
            class="tree-item child"
          >
            <text class="tree-label"
              >{{ child.name
              }}<text class="tree-type"
                >[{{ typeLabel(child.type) }}]</text
              ></text
            >
            <view class="tree-actions">
              <SvgIcon
                v-if="canEdit"
                name="edit-pen"
                @click="editMenu(child)"
              />
              <SvgIcon
                v-if="canDelete"
                name="trash"
                color="#ef4444"
                @click="delMenu(child.id)"
              />
            </view>
          </view>
        </view>
      </view>
      <view v-if="menuTree.length === 0" class="empty-tip">暂无菜单</view>
    </view>
    <FabButton v-if="canAdd" @click="editMenu(null)">
      <SvgIcon name="plus" color="#fff" size="24" />
    </FabButton>
    <Popup :show="showForm" mode="center" round @close="showForm = false">
      <view class="popup-body">
        <view class="popup-title">{{ form.id ? "编辑菜单" : "新增菜单" }}</view>
        <view class="form-row">
          <text class="form-label">类型</text>
          <radio-group @change="(e: any) => (form.type = e.detail.value)">
            <label class="radio-label"
              ><radio
                :value="MenuTypeEnum.CATALOG"
                :checked="form.type === MenuTypeEnum.CATALOG"
                color="#3b82f6"
              />目录</label
            >
            <label class="radio-label"
              ><radio
                :value="MenuTypeEnum.MENU"
                :checked="form.type === MenuTypeEnum.MENU"
                color="#3b82f6"
              />菜单</label
            >
            <label class="radio-label"
              ><radio
                :value="MenuTypeEnum.BUTTON"
                :checked="form.type === MenuTypeEnum.BUTTON"
                color="#3b82f6"
              />按钮</label
            >
            <label class="radio-label"
              ><radio
                :value="MenuTypeEnum.EXTLINK"
                :checked="form.type === MenuTypeEnum.EXTLINK"
                color="#3b82f6"
              />外链</label
            >
          </radio-group>
        </view>
        <view class="form-row">
          <text class="form-label">名称</text>
          <input
            class="form-input"
            v-model="form.name"
            placeholder="菜单名称"
          />
        </view>
        <view class="form-row">
          <text class="form-label">上级菜单</text>
          <input
            class="form-input"
            type="number"
            v-model.number="form.parentId"
            placeholder="上级菜单ID（0为顶级）"
          />
        </view>
        <view
          v-if="
            form.type === MenuTypeEnum.MENU ||
            form.type === MenuTypeEnum.EXTLINK
          "
          class="form-row"
        >
          <text class="form-label">路径</text>
          <input
            class="form-input"
            v-model="form.path"
            placeholder="路由路径"
          />
        </view>
        <view v-if="form.type === MenuTypeEnum.MENU" class="form-row">
          <text class="form-label">组件路径</text>
          <input
            class="form-input"
            v-model="form.component"
            placeholder="组件路径"
          />
        </view>
        <view v-if="form.type === MenuTypeEnum.BUTTON" class="form-row">
          <text class="form-label">权限标识</text>
          <input
            class="form-input"
            v-model="form.perm"
            placeholder="权限标识（如 sys:user:add）"
          />
        </view>
        <view class="form-row">
          <text class="form-label">排序</text>
          <input
            class="form-input"
            type="number"
            v-model.number="form.sort"
            placeholder="排序"
          />
        </view>
        <view class="form-row">
          <text class="form-label">图标</text>
          <input class="form-input" v-model="form.icon" placeholder="图标" />
        </view>
        <view class="form-row">
          <text class="form-label">可见</text>
          <switch
            :checked="form.visible === 1"
            @change="(e: any) => (form.visible = e.detail.value ? 1 : 0)"
          />
        </view>
        <button class="btn btn-primary" :loading="saving" @click="handleSave">
          保存
        </button>
      </view>
    </Popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import FabButton from "@/components/common/FabButton.vue";
import Popup from "@/components/common/Popup.vue";
import { MenuAPI, MenuTypeEnum } from "dehaze-sdk-js";
import type { MenuVO } from "dehaze-sdk-js";
import { useAuthStore } from "@/store/auth";

const authStore = useAuthStore();
const canAdd = computed(() => authStore.hasPerm("sys:menu:add"));
const canEdit = computed(() => authStore.hasPerm("sys:menu:edit"));
const canDelete = computed(() => authStore.hasPerm("sys:menu:delete"));

const menuTree = ref<MenuVO[]>([]);
const expandedIds = ref<number[]>([]);
const showForm = ref(false);
const form = ref<any>({
  name: "",
  type: MenuTypeEnum.CATALOG,
  parentId: 0,
  path: "",
  component: "",
  perm: "",
  sort: 0,
  icon: "",
  visible: 1,
});
const saving = ref(false);

const typeLabel = (type?: MenuTypeEnum) => {
  const map: Record<MenuTypeEnum, string> = {
    [MenuTypeEnum.CATALOG]: "目录",
    [MenuTypeEnum.MENU]: "菜单",
    [MenuTypeEnum.BUTTON]: "按钮",
    [MenuTypeEnum.EXTLINK]: "外链",
  };
  return (type && map[type]) || "未知";
};

const fetchTree = async () => {
  try {
    menuTree.value = await MenuAPI.getList({});
  } catch {}
};
const toggleExpand = (menu: MenuVO) => {
  if (!menu.id) return;
  const idx = expandedIds.value.indexOf(menu.id);
  if (idx > -1) expandedIds.value.splice(idx, 1);
  else expandedIds.value.push(menu.id);
};
const editMenu = (menu: MenuVO | null) => {
  if (menu) {
    form.value = {
      id: menu.id,
      name: menu.name || "",
      type: menu.type ?? MenuTypeEnum.CATALOG,
      parentId: menu.parentId ?? 0,
      path: menu.path || "",
      component: menu.component || "",
      perm: menu.perm || "",
      sort: menu.sort ?? 0,
      icon: menu.icon || "",
      visible: menu.visible ?? 1,
    };
  } else {
    form.value = {
      name: "",
      type: MenuTypeEnum.CATALOG,
      parentId: 0,
      path: "",
      component: "",
      perm: "",
      sort: 0,
      icon: "",
      visible: 1,
    };
  }
  showForm.value = true;
};
const handleSave = async () => {
  saving.value = true;
  try {
    if (form.value.id) await MenuAPI.update(String(form.value.id), form.value);
    else await MenuAPI.add(form.value);
    showForm.value = false;
    fetchTree();
    uni.showToast({ title: "保存成功", icon: "success" });
  } catch {
    uni.showToast({ title: "保存失败", icon: "error" });
  } finally {
    saving.value = false;
  }
};
const delMenu = async (id?: number) => {
  if (!id) return;
  const res = await uni.showModal({
    title: "确认删除",
    content: "确定删除该菜单吗？",
  });
  if (!res.confirm) return;
  try {
    await MenuAPI.deleteByIds(String(id));
    fetchTree();
    uni.showToast({ title: "删除成功", icon: "success" });
  } catch {
    uni.showToast({ title: "删除失败", icon: "error" });
  }
};

fetchTree();
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.tree-node {
  margin-bottom: 4rpx;
}
.tree-item {
  display: flex;
  align-items: center;
  gap: 12rpx;
  padding: 16rpx;
  border-bottom: 1rpx solid $color-border;
}
.tree-item.child {
  padding-left: 60rpx;
  background: $color-bg-secondary;
}
.tree-label {
  flex: 1;
}
.tree-type {
  color: $color-text-secondary;
  font-size: $font-xs;
}
.tree-actions {
  display: flex;
  gap: 16rpx;
}
.tree-children {
  border-left: 2rpx solid $color-border;
  margin-left: 24rpx;
}
.popup-body {
  padding: 30rpx;
  width: 90vw;
}
.popup-title {
  font-size: $font-lg;
  font-weight: bold;
  margin-bottom: 20rpx;
}
.form-row {
  display: flex;
  align-items: center;
  padding: 20rpx 0;
  border-bottom: 1rpx solid $color-border;
}
.form-label {
  width: 180rpx;
  flex-shrink: 0;
  color: $color-text-primary;
}
.form-input {
  flex: 1;
  font-size: $font-sm;
}
.radio-group {
  flex: 1;
  display: flex;
  flex-wrap: wrap;
  gap: 16rpx;
}
.radio-label {
  display: inline-flex;
  align-items: center;
  gap: 4rpx;
  font-size: $font-sm;
}
.btn {
  width: 100%;
  margin-top: 40rpx;
  padding: 16rpx 0;
  border-radius: $radius-sm;
  font-size: $font-sm;

  &::after {
    border: none;
  }
}
.btn-primary {
  background: $color-primary;
  color: $color-white;
}
</style>
