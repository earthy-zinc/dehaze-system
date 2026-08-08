<template>
  <PageLayout level="L2" title="菜单管理">
    <view class="page-body">
      <view v-for="menu in menuTree" :key="menu.id" class="tree-node">
        <view class="tree-item" @click="toggleExpand(menu)">
          <text class="tree-label"
            >{{ menu.name }}
            <text class="tree-type"
              >[{{ typeLabel(menu.type) }}]</text
            ></text
          >
          <view class="tree-actions" @click.stop>
            <SvgIcon name="edit-pen" @click="editMenu(menu)" />
            <SvgIcon
              name="trash"
              @click="delMenu(menu.id)"
              color="$color-error"
            />
          </view>
          <SvgIcon
            v-if="menu.children?.length"
            :name="expandedIds.includes(menu.id) ? 'arrow-down' : 'arrow-right'"
            size="16"
          />
        </view>
        <view
          v-if="expandedIds.includes(menu.id) && menu.children"
          class="tree-children"
        >
          <view
            v-for="child in menu.children"
            :key="child.id"
            class="tree-item child"
          >
            <text class="tree-label"
              >{{ child.name }}
              <text class="tree-type"
                >[{{ typeLabel(child.type) }}]</text
              ></text
            >
            <view class="tree-actions">
              <SvgIcon name="edit-pen" @click="editMenu(child)" />
              <SvgIcon
                name="trash"
                @click="delMenu(child.id)"
                color="$color-error"
              />
            </view>
          </view>
        </view>
      </view>
      <u-empty v-if="menuTree.length === 0" text="暂无菜单" />
    </view>
    <view class="fab-btn" @click="editMenu(null)"
      ><SvgIcon name="plus" color="#fff" size="24"
    /></view>
    <u-popup :show="showForm" @close="showForm = false" round>
      <view class="popup-content">
        <view class="popup-title">{{ form.id ? "编辑菜单" : "新增菜单" }}</view>
        <u-form :model="form">
          <u-form-item label="类型">
            <u-radio-group v-model="form.type">
              <u-radio :label="0" name="目录" />
              <u-radio :label="1" name="菜单" />
              <u-radio :label="2" name="按钮" />
              <u-radio :label="3" name="外链" />
            </u-radio-group>
          </u-form-item>
          <u-form-item label="名称"
            ><u-input v-model="form.name" placeholder="菜单名称"
          /></u-form-item>
          <u-form-item label="上级菜单">
            <u-input
              v-model.number="form.parentId"
              type="number"
              placeholder="上级菜单ID（0为顶级）"
            />
          </u-form-item>
          <u-form-item v-if="form.type === 1 || form.type === 3" label="路径"
            ><u-input v-model="form.path" placeholder="路由路径"
          /></u-form-item>
          <u-form-item v-if="form.type === 1" label="组件路径"
            ><u-input v-model="form.component" placeholder="组件路径"
          /></u-form-item>
          <u-form-item v-if="form.type === 2" label="权限标识"
            ><u-input v-model="form.perm" placeholder="权限标识（如 sys:user:add）"
          /></u-form-item>
          <u-form-item label="排序"
            ><u-input
              v-model.number="form.sort"
              type="number"
              placeholder="排序"
          /></u-form-item>
          <u-form-item label="图标"
            ><u-input v-model="form.icon" placeholder="图标"
          /></u-form-item>
          <u-form-item label="可见">
            <u-switch
              :checked="form.visible === 1"
              @change="(val: boolean) => (form.visible = val ? 1 : 0)"
            />
          </u-form-item>
        </u-form>
        <u-button type="primary" @click="handleSave" :loading="saving"
          >保存</u-button
        >
      </view>
    </u-popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { MenuAPI } from "dehaze-sdk-js";
import type { MenuVO } from "dehaze-sdk-js";

const menuTree = ref<MenuVO[]>([]);
const expandedIds = ref<number[]>([]);
const showForm = ref(false);
const form = ref<any>({
  name: "",
  type: 0,
  parentId: 0,
  path: "",
  component: "",
  perm: "",
  sort: 0,
  icon: "",
  visible: 1,
});
const saving = ref(false);

const typeLabel = (type?: number) => {
  const map: Record<number, string> = { 0: "目录", 1: "菜单", 2: "按钮", 3: "外链" };
  return map[type ?? 0] || "未知";
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
      type: menu.type ?? 0,
      parentId: menu.parentId ?? 0,
      path: menu.routePath || "",
      component: menu.component || "",
      perm: menu.perm || "",
      sort: menu.sort ?? 0,
      icon: menu.icon || "",
      visible: menu.visible ?? 1,
    };
  } else {
    form.value = {
      name: "",
      type: 0,
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
  font-size: 24rpx;
}
.tree-actions {
  display: flex;
  gap: 16rpx;
}
.tree-children {
  border-left: 2rpx solid $color-border;
  margin-left: 24rpx;
}
.fab-btn {
  position: fixed;
  right: 40rpx;
  bottom: 100rpx;
  width: 96rpx;
  height: 96rpx;
  border-radius: 50%;
  background: $color-primary;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.2);
  z-index: 99;
}
.popup-content {
  padding: 30rpx;
  width: 90vw;
}
.popup-title {
  font-size: 32rpx;
  font-weight: bold;
  margin-bottom: 20rpx;
}
</style>
