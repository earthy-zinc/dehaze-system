<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="角色名称"
            @keyup.enter="handleQuery"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleQuery"
            ><el-icon><Search /></el-icon>搜索</el-button
          >
          <el-button @click="resetQuery"><el-icon><Refresh /></el-icon>重置</el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <el-button
          v-hasPerm="['sys:role:add']"
          type="success"
          @click="openDialog()"
          ><el-icon><Plus /></el-icon>新增</el-button
        >
        <el-button
          v-hasPerm="['sys:role:delete']"
          :disabled="ids.length === 0"
          type="danger"
          @click="handleDelete()"
          ><el-icon><Delete /></el-icon>删除</el-button
        >
      </template>

      <el-table
        ref="dataTableRef"
        v-loading="loading"
        :data="roleList"
        border
        highlight-current-row
        @selection-change="handleSelectionChange"
      >
        <el-table-column align="center" type="selection" width="55" />
        <el-table-column label="角色名称" min-width="100" prop="name" />
        <el-table-column label="角色编码" prop="code" width="150" />

        <el-table-column align="center" label="状态" width="100">
          <template #default="scope">
            <el-switch
              v-model="scope.row.status"
              :active-value="1"
              :inactive-value="0"
              @change="handleStatusChange(scope.row)"
            />
          </template>
        </el-table-column>

        <el-table-column
          align="center"
          label="数据权限"
          prop="dataScopeLabel"
          width="120"
        />

        <el-table-column align="center" label="排序" prop="sort" width="80" />

        <el-table-column
          align="center"
          label="创建时间"
          prop="createTime"
          width="180"
        />

        <el-table-column fixed="right" label="操作" width="220">
          <template #default="scope">
            <el-button
              v-hasPerm="['sys:role:edit']"
              link
              size="small"
              type="primary"
              @click="openMenuDialog(scope.row)"
            >
              <el-icon><Position /></el-icon>分配权限
            </el-button>
            <el-button
              v-hasPerm="['sys:role:edit']"
              link
              size="small"
              type="primary"
              @click="openDialog(scope.row.id)"
            >
              <el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              v-hasPerm="['sys:role:delete']"
              link
              size="small"
              type="primary"
              @click="handleDelete(scope.row)"
            >
              <el-icon><Delete /></el-icon>删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <pagination
        v-if="total > 0"
        v-model:limit="queryParams.pageSize"
        v-model:page="queryParams.pageNum"
        v-model:total="total"
        @pagination="handleQuery"
      />
    </el-card>

    <!-- 角色表单弹窗 -->
    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      width="500px"
      @close="closeDialog"
    >
      <el-form
        ref="roleFormRef"
        :model="formData"
        :rules="rules"
        label-width="100px"
      >
        <el-form-item label="角色名称" prop="name">
          <el-input v-model="formData.name" placeholder="请输入角色名称" />
        </el-form-item>

        <el-form-item label="角色编码" prop="code">
          <el-input
            v-model="formData.code"
            :readonly="!!formData.id"
            placeholder="请输入角色编码"
          />
        </el-form-item>

        <el-form-item label="数据权限" prop="dataScope">
          <el-select v-model="formData.dataScope">
            <el-option :key="0" :value="0" label="全部数据" />
            <el-option :key="1" :value="1" label="部门及子部门数据" />
            <el-option :key="2" :value="2" label="本部门数据" />
            <el-option :key="3" :value="3" label="本人数据" />
          </el-select>
        </el-form-item>

        <el-form-item label="状态" prop="status">
          <el-radio-group v-model="formData.status">
            <el-radio :label="1">正常</el-radio>
            <el-radio :label="0">停用</el-radio>
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
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleSubmit">确 定</el-button>
          <el-button @click="closeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 分配菜单弹窗  -->
    <el-dialog
      v-model="menuDialogVisible"
      :title="'【' + checkedRole.name + '】权限分配'"
      width="800px"
    >
      <div style="margin-bottom: 12px">
        <el-button size="small" @click="handleToggleCheckAll">
          {{ isCheckAll ? "取消全选" : "全选" }}
        </el-button>
        <el-button size="small" @click="handleToggleExpandAll">
          {{ isExpandAll ? "收起所有" : "展开所有" }}
        </el-button>
      </div>

      <el-scrollbar v-loading="loading" max-height="600px">
        <el-tree
          ref="menuRef"
          :data="menuList"
          :default-expand-all="isExpandAll"
          node-key="value"
          show-checkbox
        >
          <template #default="{ data }">
            {{ data.label }}
          </template>
        </el-tree>
      </el-scrollbar>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleRoleMenuSubmit"
            >确 定</el-button
          >
          <el-button @click="menuDialogVisible = false">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  MenuAPI,
  OptionType,
  RoleAPI,
  RoleForm,
  RolePageVO,
  RoleQuery,
} from "dehaze-sdk-js";
import {
  Delete,
  Edit,
  Plus,
  Position,
  Refresh,
  Search,
} from "@element-plus/icons-vue";

defineOptions({
  name: "Role",
  inheritAttrs: false,
});

const queryFormRef = ref(ElForm);
const roleFormRef = ref(ElForm);
const menuRef = ref(ElTree);

const loading = ref(false);
const ids = ref<number[]>([]);
const total = ref(0);

const queryParams = reactive<RoleQuery>({
  pageNum: 1,
  pageSize: 10,
});

const roleList = ref<RolePageVO[]>();

const dialog = reactive({
  title: "",
  visible: false,
});

const formData = reactive<RoleForm>({
  sort: 1,
  status: 1,
  dataScope: 2,
  code: "",
  name: "",
});

const rules = reactive({
  name: [{ required: true, message: "请输入角色名称", trigger: "blur" }],
  code: [
    { required: true, message: "请输入角色编码", trigger: "blur" },
    {
      pattern: /^[A-Z_]+$/,
      message: "角色编码只能包含大写字母和下划线",
      trigger: "blur",
    },
  ],
  dataScope: [{ required: true, message: "请选择数据权限", trigger: "blur" }],
  status: [{ required: true, message: "请选择状态", trigger: "blur" }],
});

const menuDialogVisible = ref(false);

const menuList = ref<OptionType[]>([]);

const isCheckAll = ref(false);
const isExpandAll = ref(true);

interface CheckedRole {
  id?: number;
  name?: string;
}
let checkedRole: CheckedRole = reactive({});

/** 查询 */
function handleQuery() {
  loading.value = true;
  RoleAPI.getPage(queryParams)
    .then((data) => {
      roleList.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}
/** 重置查询 */
function resetQuery() {
  queryFormRef.value.resetFields();
  queryParams.pageNum = 1;
  handleQuery();
}

/** 行checkbox 选中事件 */
function handleSelectionChange(selection: any) {
  ids.value = selection.map((item: any) => item.id);
}

/** 切换角色状态 */
function handleStatusChange(row: RolePageVO) {
  const roleId = row.id;
  if (!roleId || row.status === undefined) return;
  const text = row.status === 1 ? "启用" : "禁用";
  RoleAPI.updateStatus(roleId, row.status)
    .then(() => {
      ElMessage.success(`${text}成功`);
    })
    .catch(() => {
      row.status = row.status === 1 ? 0 : 1;
    });
}

/** 打开角色表单弹窗 */
function openDialog(roleId?: number) {
  dialog.visible = true;
  if (roleId) {
    dialog.title = "修改角色";
    RoleAPI.getFormData(roleId).then((data) => {
      Object.assign(formData, data);
    });
  } else {
    dialog.title = "新增角色";
  }
}

/** 角色保存提交 */
function handleSubmit() {
  roleFormRef.value.validate((valid: any) => {
    if (valid) {
      loading.value = true;
      const roleId = formData.id;
      if (roleId) {
        RoleAPI.update(roleId, formData)
          .then(() => {
            ElMessage.success("修改成功");
            closeDialog();
            resetQuery();
          })
          .finally(() => (loading.value = false));
      } else {
        RoleAPI.add(formData)
          .then(() => {
            ElMessage.success("新增成功");
            closeDialog();
            resetQuery();
          })
          .finally(() => (loading.value = false));
      }
    }
  });
}

/** 关闭表单弹窗 */
function closeDialog() {
  dialog.visible = false;
  resetForm();
}

/** 重置表单 */
function resetForm() {
  roleFormRef.value.resetFields();
  roleFormRef.value.clearValidate();

  formData.id = undefined;
  formData.sort = 1;
  formData.status = 1;
  formData.dataScope = 2;
}

/** 删除角色 */
function handleDelete(row?: RolePageVO) {
  const roleIds = row?.id ? String(row.id) : ids.value.join(",");
  if (!roleIds) {
    ElMessage.warning("请勾选删除项");
    return;
  }

  const confirmText = row
    ? `确认删除角色「${row.name}」吗？删除后不可恢复。`
    : "确认删除选中的角色吗？删除后不可恢复。";

  ElMessageBox.confirm(confirmText, "警告", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  }).then(() => {
    loading.value = true;
    RoleAPI.deleteByIds(roleIds)
      .then(() => {
        ElMessage.success("删除成功");
        resetQuery();
      })
      .finally(() => (loading.value = false));
  });
}

/** 打开分配菜单弹窗 */
async function openMenuDialog(row: RolePageVO) {
  const roleId = row.id;
  if (roleId) {
    checkedRole = {
      id: roleId,
      name: row.name,
    };
    menuDialogVisible.value = true;
    loading.value = true;
    isCheckAll.value = false;
    isExpandAll.value = true;

    // 获取所有的菜单
    menuList.value = await MenuAPI.getOptions();

    // 回显角色已拥有的菜单
    RoleAPI.getRoleMenuIds(roleId)
      .then((data) => {
        const checkedMenuIds = data;
        checkedMenuIds.forEach((menuId) =>
          menuRef.value.setChecked(menuId, true, false)
        );
      })
      .finally(() => {
        loading.value = false;
      });
  }
}

/** 全选/取消全选 */
function handleToggleCheckAll() {
  if (isCheckAll.value) {
    menuRef.value.setCheckedKeys([]);
    isCheckAll.value = false;
  } else {
    const allKeys: (string | number)[] = [];
    const walk = (nodes: OptionType[]) => {
      nodes.forEach((n) => {
        allKeys.push(n.value);
        if (n.children?.length) walk(n.children);
      });
    };
    walk(menuList.value);
    menuRef.value.setCheckedKeys(allKeys);
    isCheckAll.value = true;
  }
}

/** 展开/收起所有 */
function handleToggleExpandAll() {
  isExpandAll.value = !isExpandAll.value;
  const tree = menuRef.value as any;
  tree.store.nodesAll.forEach((node: any) => {
    node.expanded = isExpandAll.value;
  });
}

/** 角色分配菜单保存提交 */
function handleRoleMenuSubmit() {
  const roleId = checkedRole.id;
  if (roleId) {
    const checkedMenuIds: number[] = menuRef.value
      .getCheckedNodes(false, true)
      .map((node: any) => node.value);

    loading.value = true;
    RoleAPI.updateRoleMenus(roleId, checkedMenuIds)
      .then(() => {
        ElMessage.success("分配权限成功");
        menuDialogVisible.value = false;
        resetQuery();
      })
      .finally(() => {
        loading.value = false;
      });
  }
}

onMounted(() => {
  handleQuery();
});
</script>
