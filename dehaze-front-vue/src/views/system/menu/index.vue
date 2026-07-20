<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="菜单名称"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery"
            ><template #icon><el-icon><Search /></el-icon></template>搜索</el-button
          >
          <el-button @click="resetQuery">
            <template #icon><el-icon><Refresh /></el-icon></template>
            重置</el-button
          >
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <el-button
          v-hasPerm="['sys:menu:add']"
          type="success"
          @click="openDialog(0)"
        >
          <template #icon><el-icon><Plus /></el-icon></template>
          新增</el-button
        >
      </template>

      <el-table
        v-loading="loading"
        :data="menuList"
        :expand-row-keys="['1']"
        :tree-props="{
          children: 'children',
          hasChildren: 'hasChildren',
        }"
        highlight-current-row
        row-key="id"
        @row-click="onRowClick"
      >
        <el-table-column label="菜单名称" min-width="200">
          <template #default="scope">
            <svg-icon :icon-class="scope.row.icon" />
            {{ scope.row.name }}
          </template>
        </el-table-column>

        <el-table-column align="center" label="类型" width="80">
          <template #default="scope">
            <el-tag
              v-if="scope.row.type === MenuTypeEnum.CATALOG"
              type="warning"
              >目录</el-tag
            >
            <el-tag v-if="scope.row.type === MenuTypeEnum.MENU" type="success"
              >菜单</el-tag
            >
            <el-tag v-if="scope.row.type === MenuTypeEnum.BUTTON" type="danger"
              >按钮</el-tag
            >
            <el-tag v-if="scope.row.type === MenuTypeEnum.EXTLINK" type="info"
              >外链</el-tag
            >
          </template>
        </el-table-column>

        <el-table-column
          align="left"
          label="路由路径"
          prop="path"
          width="150"
        />

        <el-table-column
          align="left"
          label="组件路径"
          prop="component"
          width="250"
        />

        <el-table-column
          align="center"
          label="权限标识"
          prop="perm"
          width="200"
        />

        <el-table-column align="center" label="状态" width="80">
          <template #default="scope">
            <el-switch
              v-model="scope.row.visible"
              :active-value="1"
              :inactive-value="0"
              @change="(val) => handleVisibleChange(scope.row, val)"
            />
          </template>
        </el-table-column>

        <el-table-column align="center" label="排序" prop="sort" width="80" />

        <el-table-column align="center" fixed="right" label="操作" width="220">
          <template #default="scope">
            <el-button
              v-if="scope.row.type == 'CATALOG' || scope.row.type == 'MENU'"
              v-hasPerm="['sys:menu:add']"
              link
              size="small"
              type="primary"
              @click.stop="openDialog(scope.row.id)"
            >
              <el-icon><Plus /></el-icon>新增
            </el-button>

            <el-button
              v-hasPerm="['sys:menu:edit']"
              link
              size="small"
              type="primary"
              @click.stop="openDialog(undefined, scope.row.id)"
            >
              <el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              v-hasPerm="['sys:menu:delete']"
              link
              size="small"
              type="primary"
              @click.stop="handleDelete(scope.row)"
              ><el-icon><Delete /></el-icon>
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      append-to-body
      destroy-on-close
      top="5vh"
      width="1000px"
      @close="closeDialog"
    >
      <el-form
        ref="menuFormRef"
        :model="formData"
        :rules="rules"
        label-width="160px"
      >
        <el-form-item label="父级菜单" prop="parentId">
          <el-tree-select
            v-model="formData.parentId"
            :data="menuOptions"
            :render-after-expand="false"
            check-strictly
            filterable
            placeholder="选择上级菜单"
          />
        </el-form-item>

        <el-form-item label="菜单名称" prop="name">
          <el-input v-model="formData.name" placeholder="请输入菜单名称" />
        </el-form-item>

        <el-form-item label="菜单类型" prop="type">
          <el-radio-group v-model="formData.type" @change="onMenuTypeChange">
            <el-radio label="CATALOG">目录</el-radio>
            <el-radio label="MENU">菜单</el-radio>
            <el-radio label="BUTTON">按钮</el-radio>
            <el-radio label="EXTLINK">外链</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item
          v-if="formData.type == 'EXTLINK'"
          label="外链地址"
          prop="path"
        >
          <el-input v-model="formData.path" placeholder="请输入外链完整路径" />
        </el-form-item>

        <el-form-item
          v-if="
            formData.type == MenuTypeEnum.CATALOG ||
            formData.type == MenuTypeEnum.MENU
          "
          label="路由路径"
          prop="path"
        >
          <el-input
            v-if="formData.type == MenuTypeEnum.CATALOG"
            v-model="formData.path"
            placeholder="system"
          />
          <el-input v-else v-model="formData.path" placeholder="user" />
        </el-form-item>

        <!-- 组件页面完整路径 -->
        <el-form-item
          v-if="formData.type == MenuTypeEnum.MENU"
          label="页面路径"
          prop="component"
        >
          <el-input
            v-model="formData.component"
            placeholder="system/user/index"
            style="width: 95%"
          >
            <template v-if="formData.type == MenuTypeEnum.MENU" #prepend
              >src/views/</template
            >
            <template v-if="formData.type == MenuTypeEnum.MENU" #append
              >.vue</template
            >
          </el-input>
        </el-form-item>

        <el-form-item
          v-if="formData.type !== MenuTypeEnum.BUTTON"
          label="显示状态"
          prop="visible"
        >
          <el-radio-group v-model="formData.visible">
            <el-radio :label="1">显示</el-radio>
            <el-radio :label="0">隐藏</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item
          v-if="formData.type === MenuTypeEnum.CATALOG"
          label="根目录始终显示"
        >
          <template #label>
            <div>
              根目录始终显示
              <el-tooltip effect="light" placement="bottom">
                <template #content
                  >是：根目录只有一个子路由显示目录
                  <br />否：根目录只有一个子路由不显示目录，只显示子路由
                </template>
                <el-icon class="inline-block"><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
          </template>

          <el-radio-group v-model="formData.alwaysShow">
            <el-radio :label="1">是</el-radio>
            <el-radio :label="0">否</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item
          v-if="formData.type === MenuTypeEnum.MENU"
          label="是否缓存"
        >
          <el-radio-group v-model="formData.keepAlive">
            <el-radio :label="1">是</el-radio>
            <el-radio :label="0">否</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item label="排序" prop="sort">
          <el-input-number
            v-model="formData.sort"
            :min="0"
            controls-position="right"
            style="width: 100px"
          />
        </el-form-item>

        <!-- 权限标识 -->
        <el-form-item
          v-if="formData.type == MenuTypeEnum.BUTTON"
          label="权限标识"
          prop="perm"
        >
          <el-input v-model="formData.perm" placeholder="sys:user:add" />
        </el-form-item>

        <el-form-item
          v-if="formData.type !== MenuTypeEnum.BUTTON"
          label="图标"
          prop="icon"
        >
          <!-- 图标选择器 -->
          <icon-select v-model="formData.icon" />
        </el-form-item>

        <el-form-item
          v-if="formData.type == MenuTypeEnum.CATALOG"
          label="跳转路由"
        >
          <el-input v-model="formData.redirect" placeholder="跳转路由" />
        </el-form-item>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="submitForm">确 定</el-button>
          <el-button @click="closeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
defineOptions({
  name: "Menu",
  inheritAttrs: false,
});

import {
  MenuAPI,
  MenuForm,
  MenuQuery,
  MenuTypeEnum,
  MenuVO,
} from "dehaze-sdk-js";
import {
  Delete,
  Edit,
  Plus,
  QuestionFilled,
  Refresh,
  Search,
} from "@element-plus/icons-vue";
import { usePermissionStoreHook } from "@/store/modules/permission";
import { useUserStore } from "@/store";

const queryFormRef = ref(ElForm);
const menuFormRef = ref(ElForm);

const loading = ref(false);
const dialog = reactive({
  title: "",
  visible: false,
});

const queryParams = reactive<MenuQuery>({});
const menuList = ref<MenuVO[]>([]);

const menuOptions = ref<OptionType[]>([]);

const formData = reactive<MenuForm>({
  parentId: 0,
  visible: 1,
  sort: 1,
  type: MenuTypeEnum.CATALOG,
  alwaysShow: 0,
  keepAlive: 0,
});

const userStore = useUserStore();

const rules = computed(() => {
  const r: Record<string, any> = {
    parentId: [{ required: true, message: "请选择顶级菜单", trigger: "blur" }],
    name: [
      { required: true, message: "请输入菜单名称", trigger: "blur" },
      { min: 2, max: 64, message: "长度在 2 到 64 个字符", trigger: "blur" },
    ],
    type: [{ required: true, message: "请选择菜单类型", trigger: "blur" }],
  };

  if (formData.type === MenuTypeEnum.MENU) {
    r.path = [
      { required: true, message: "请输入路由路径", trigger: "blur" },
      { pattern: /^\//, message: "路由路径需以 / 开头", trigger: "blur" },
    ];
    r.component = [
      { required: true, message: "请输入组件路径", trigger: "blur" },
    ];
  } else if (formData.type === MenuTypeEnum.BUTTON) {
    r.perm = [
      { required: true, message: "请输入权限标识", trigger: "blur" },
      {
        pattern: /^[a-z]+:[a-z]+:[a-z]+$/,
        message: "权限标识格式为 xxx:xxx:xxx",
        trigger: "blur",
      },
    ];
  } else if (formData.type === MenuTypeEnum.EXTLINK) {
    r.path = [
      { required: true, message: "请输入外链地址", trigger: "blur" },
      { type: "url", message: "请输入正确的URL格式", trigger: "blur" },
    ];
  }

  return r;
});

// 选择表格的行菜单ID
const selectedRowMenuId = ref<number | undefined>();

const menuCacheData = reactive({
  type: "",
  path: "",
});

/**
 * 查询
 */
function handleQuery() {
  // 重置父组件
  loading.value = true;
  MenuAPI.getList(queryParams)
    .then((data) => {
      menuList.value = data;
    })
    .then(() => {
      loading.value = false;
    });
}

/** 重置查询 */
function resetQuery() {
  queryFormRef.value.resetFields();
  handleQuery();
}

/**行点击事件 */
function onRowClick(row: MenuVO) {
  selectedRowMenuId.value = row.id;
}

/**
 * 打开表单弹窗
 *
 * @param parentId 父菜单ID
 * @param menuId 菜单ID
 */
function openDialog(parentId?: number, menuId?: number) {
  MenuAPI.getOptions()
    .then((data) => {
      menuOptions.value = [{ value: 0, label: "顶级菜单", children: data }];
    })
    .then(() => {
      dialog.visible = true;
      if (menuId) {
        dialog.title = "编辑菜单";
        MenuAPI.getFormData(menuId).then((data) => {
          Object.assign(formData, data);
          menuCacheData.type = data.type;
          menuCacheData.path = data.path ?? "";
        });
      } else {
        dialog.title = "新增菜单";
        formData.parentId = parentId;
      }
    });
}

/** 菜单类型切换事件处理 */
function onMenuTypeChange() {
  // 如果菜单类型改变，清空路由路径；未改变在切换后还原路由路径
  if (formData.type !== menuCacheData.type) {
    formData.path = "";
  } else {
    formData.path = menuCacheData.path;
  }
}

/** 菜单保存提交 */
function submitForm() {
  menuFormRef.value.validate((isValid: boolean) => {
    if (isValid) {
      const menuId = formData.id;
      if (menuId) {
        MenuAPI.update(menuId, formData).then(() => {
          ElMessage.success("修改成功");
          usePermissionStoreHook().generateRoutes(userStore.user.roles);
          closeDialog();
          handleQuery();
        });
      } else {
        MenuAPI.add(formData).then(() => {
          ElMessage.success("新增成功");
          usePermissionStoreHook().generateRoutes(userStore.user.roles);
          closeDialog();
          handleQuery();
        });
      }
    }
  });
}

/** 删除菜单 */
function handleDelete(row: MenuVO) {
  if (!row.id) {
    ElMessage.warning("请勾选删除项");
    return false;
  }
  const menuId = row.id;

  ElMessageBox.confirm(
    `确认删除菜单「${row.name}」吗？删除后不可恢复。`,
    "警告",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    }
  )
    .then(() => {
      MenuAPI.deleteById(menuId).then(() => {
        ElMessage.success("删除成功");
        usePermissionStoreHook().generateRoutes(userStore.user.roles);
        handleQuery();
      });
    })
    .catch(() => ElMessage.info("已取消删除"));
}

/** 显示状态切换 */
function handleVisibleChange(row: MenuVO, val: string | number | boolean) {
  const visibleVal = Number(val);
  const oldVal = visibleVal === 1 ? 0 : 1;
  ElMessageBox.confirm(
    `确认${visibleVal === 1 ? "显示" : "隐藏"}菜单「${row.name}」吗？`,
    "提示",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    }
  )
    .then(() => {
      MenuAPI.update(String(row.id), {
        ...row,
        visible: visibleVal,
      } as MenuForm).then(() => {
        ElMessage.success("修改成功");
        usePermissionStoreHook().generateRoutes(userStore.user.roles);
        handleQuery();
      });
    })
    .catch(() => {
      row.visible = oldVal;
      ElMessage.info("已取消");
    });
}

/** 关闭弹窗 */
function closeDialog() {
  dialog.visible = false;
  resetForm();
}

/** 重置表单 */
function resetForm() {
  menuFormRef.value.resetFields();
  menuFormRef.value.clearValidate();

  formData.id = undefined;
  formData.parentId = 0;
  formData.visible = 1;
  formData.sort = 1;
  formData.type = MenuTypeEnum.CATALOG;
  formData.perm = undefined;
  formData.component = undefined;
  formData.path = undefined;
  formData.redirect = undefined;
  formData.alwaysShow = undefined;
  formData.keepAlive = undefined;
}

onMounted(() => {
  handleQuery();
});
</script>
