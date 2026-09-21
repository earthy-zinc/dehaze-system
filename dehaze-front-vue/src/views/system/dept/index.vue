<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            placeholder="部门名称"
            @keyup.enter="handleQuery"
          />
        </el-form-item>

        <el-form-item label="部门状态" prop="status">
          <el-select
            v-model="queryParams.status"
            class="!w-[100px]"
            clearable
            placeholder="全部"
          >
            <el-option :value="1" label="启用" />
            <el-option :value="0" label="禁用" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button class="filter-item" type="primary" @click="handleQuery">
            <el-icon><Search /></el-icon>
            搜索
          </el-button>
          <el-button @click="resetQuery">
            <el-icon>
              <Refresh />
            </el-icon>
            重置
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <div>
            <el-button
              v-hasPerm="['sys:dept:add']"
              type="success"
              @click="openDialog(0, undefined)"
              ><el-icon><Plus /></el-icon>新增</el-button
            >
            <el-button
              v-hasPerm="['sys:dept:delete']"
              :disabled="ids.length === 0"
              type="danger"
              @click="handleDelete()"
            >
              <el-icon>
                <Delete />
              </el-icon>
              删除
            </el-button>
          </div>
          <ImportExportToolbar
            module="dept"
            :query-params="queryParams"
            @import-complete="handleQuery"
          />
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="deptList"
        :tree-props="{
          children: 'children',
          hasChildren: 'hasChildren',
        }"
        default-expand-all
        row-key="id"
        @selection-change="handleSelectionChange"
      >
        <el-table-column align="center" type="selection" width="55" />
        <el-table-column label="部门名称" min-width="200" prop="name" />
        <el-table-column label="状态" prop="status" width="100">
          <template #default="scope">
            <el-switch
              v-model="scope.row.status"
              :active-value="1"
              :inactive-value="0"
              @change="handleStatusChange(scope.row)"
            />
          </template>
        </el-table-column>

        <el-table-column label="排序" prop="sort" width="100" />

        <el-table-column label="创建时间" prop="createTime" width="180" />

        <el-table-column align="left" fixed="right" label="操作" width="260">
          <template #default="scope">
            <el-button
              v-hasPerm="['sys:dept:add']"
              :disabled="isMaxLevel(scope.row)"
              link
              size="small"
              type="primary"
              @click.stop="openDialog(scope.row.id, undefined)"
              ><el-icon><Plus /></el-icon>新增下级
            </el-button>
            <el-button
              v-hasPerm="['sys:dept:edit']"
              link
              size="small"
              type="primary"
              @click.stop="openDialog(scope.row.parentId, scope.row.id)"
              ><el-icon><Edit /></el-icon>编辑
            </el-button>
            <el-button
              v-hasPerm="['sys:dept:delete']"
              :disabled="scope.row.id === 1"
              link
              size="small"
              type="primary"
              @click.stop="handleDelete(scope.row)"
            >
              <el-icon><Delete /></el-icon>删除
            </el-button>
            <el-button
              link
              size="small"
              type="primary"
              @click.stop="openUserDialog(scope.row)"
            >
              <el-icon><User /></el-icon>用户
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      width="600px"
      @closed="closeDialog"
    >
      <el-form
        ref="deptFormRef"
        :model="formData"
        :rules="rules"
        label-width="80px"
      >
        <el-form-item label="上级部门" prop="parentId">
          <el-tree-select
            v-model="formData.parentId"
            :data="deptOptions"
            :disabled="formData.id === 1"
            :render-after-expand="false"
            check-strictly
            filterable
            placeholder="选择上级部门"
          />
        </el-form-item>
        <el-form-item label="部门名称" prop="name">
          <el-input v-model="formData.name" placeholder="请输入部门名称" />
        </el-form-item>
        <el-form-item label="显示排序" prop="sort">
          <el-input-number
            v-model="formData.sort"
            :min="0"
            controls-position="right"
            style="width: 100px"
          />
        </el-form-item>
        <el-form-item label="部门状态">
          <el-radio-group v-model="formData.status">
            <el-radio :label="1">启用</el-radio>
            <el-radio :label="0">禁用</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleSubmit"> 确 定 </el-button>
          <el-button @click="closeDialog"> 取 消 </el-button>
        </div>
      </template>
    </el-dialog>

    <el-dialog
      v-model="userDialog.visible"
      :title="userDialog.title"
      width="600px"
    >
      <el-table v-loading="userDialog.loading" :data="userList">
        <el-table-column label="用户名" prop="username" min-width="120" />
        <el-table-column label="昵称" prop="nickname" min-width="120" />
        <el-table-column label="状态" prop="status" width="80">
          <template #default="scope">
            <el-tag v-if="scope.row.status === 1" type="success">启用</el-tag>
            <el-tag v-else type="info">禁用</el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
defineOptions({
  name: "SystemDept",
  inheritAttrs: false,
});

import {
  DeptAPI,
  DeptForm,
  DeptQuery,
  DeptVO,
  OptionType,
  UserAPI,
  UserPageVO,
} from "dehaze-sdk-js";
import {
  Delete,
  Edit,
  Plus,
  Refresh,
  Search,
  User,
} from "@element-plus/icons-vue";
import ImportExportToolbar from "@/components/ImportExportToolbar/index.vue";

const queryFormRef = ref(ElForm);
const deptFormRef = ref(ElForm);

const loading = ref(false);
const ids = ref<number[]>([]);
const dialog = reactive({
  title: "",
  visible: false,
});

const queryParams = reactive<DeptQuery>({});
const deptList = ref<DeptVO[]>();

const deptOptions = ref<OptionType[]>();

const userDialog = reactive({
  title: "",
  visible: false,
  loading: false,
});
const userList = ref<UserPageVO[]>([]);

const formData = reactive<DeptForm>({
  name: "",
  status: 1,
  parentId: 0,
  sort: 1,
});

const rules = reactive({
  parentId: [
    { required: true, message: "上级部门不能为空", trigger: "change" },
  ],
  name: [
    { required: true, message: "部门名称不能为空", trigger: "blur" },
    { max: 64, message: "部门名称长度不能超过 64 个字符", trigger: "blur" },
  ],
  sort: [{ required: true, message: "显示排序不能为空", trigger: "blur" }],
});

/** 查询 */
function handleQuery() {
  loading.value = true;
  DeptAPI.getList(queryParams)
    .then((data) => {
      deptList.value = data;
    })
    .finally(() => (loading.value = false));
}

/**重置查询 */
function resetQuery() {
  queryFormRef.value.resetFields();
  handleQuery();
}

/** 行复选框选中记录选中ID集合 */
function handleSelectionChange(selection: DeptVO[]) {
  ids.value = selection.map((item) => item.id!);
}

/** 获取部门下拉数据  */
async function loadDeptOptions() {
  const data = await DeptAPI.getOptions();
  deptOptions.value = [
    {
      value: 0,
      label: "顶级部门",
      children: data,
    },
  ];
}

/** 过滤指定部门及其子部门，防止选择自身或下级作为上级（循环引用） */
function filterDeptOptions(
  options: OptionType[],
  excludeId: number
): OptionType[] {
  return options
    .filter((option) => option.value !== excludeId)
    .map((option) => ({
      ...option,
      children: option.children
        ? filterDeptOptions(option.children, excludeId)
        : option.children,
    }));
}

/** 计算部门在树中的层级（根级为 1） */
function findDeptDepth(
  list: DeptVO[],
  targetId: number,
  depth = 1
): number | null {
  for (const dept of list) {
    if (dept.id === targetId) {
      return depth;
    }
    if (dept.children?.length) {
      const found = findDeptDepth(dept.children, targetId, depth + 1);
      if (found !== null) {
        return found;
      }
    }
  }
  return null;
}

/** 部门已达 5 级上限时禁止新增下级 */
function isMaxLevel(row: DeptVO): boolean {
  const depth = row.id ? findDeptDepth(deptList.value ?? [], row.id) : null;
  return depth !== null && depth >= 5;
}

/**
 * 打开弹窗
 *
 * @param parentId 父部门ID
 * @param deptId 部门ID
 */
async function openDialog(parentId?: number, deptId?: number) {
  await loadDeptOptions();
  if (deptId) {
    dialog.title = "修改部门";
    const data = await DeptAPI.getFormData(deptId);
    Object.assign(formData, data);
    deptOptions.value = filterDeptOptions(deptOptions.value ?? [], data.id!);
    dialog.visible = true;
  } else {
    dialog.title = "新增部门";
    formData.parentId = parentId ?? 0;
    dialog.visible = true;
  }
}

/** 表单提交 */
function handleSubmit() {
  deptFormRef.value.validate((valid: any) => {
    if (valid) {
      const deptId = formData.id;
      loading.value = true;
      if (deptId) {
        DeptAPI.update(deptId, formData)
          .then(() => {
            ElMessage.success("修改成功");
            closeDialog();
            handleQuery();
          })
          .finally(() => (loading.value = false));
      } else {
        DeptAPI.add(formData)
          .then(() => {
            ElMessage.success("新增成功");
            closeDialog();
            handleQuery();
          })
          .finally(() => (loading.value = false));
      }
    }
  });
}

/** 切换部门状态（禁用不级联子部门，由后端兜底校验） */
function handleStatusChange(row: DeptVO) {
  if (
    !row.id ||
    row.parentId === undefined ||
    !row.name ||
    row.sort === undefined ||
    row.status === undefined
  ) {
    return;
  }
  const text = row.status === 1 ? "启用" : "禁用";
  DeptAPI.update(row.id, {
    id: row.id,
    parentId: row.parentId,
    name: row.name,
    sort: row.sort,
    status: row.status,
  })
    .then(() => {
      ElMessage.success(`${text}成功`);
    })
    .catch(() => {
      row.status = row.status === 1 ? 0 : 1;
    });
}

/** 查看部门下关联用户 */
function openUserDialog(row: DeptVO) {
  if (!row.id) return;
  userDialog.title = `部门用户 - ${row.name}`;
  userDialog.visible = true;
  userDialog.loading = true;
  UserAPI.getPage({ deptId: row.id, pageNum: 1, pageSize: 100 })
    .then((data) => {
      userList.value = data.list;
    })
    .finally(() => (userDialog.loading = false));
}

/** 删除部门 */
function handleDelete(row?: DeptVO) {
  if (row) {
    // 单个删除
    ElMessageBox.confirm(
      `确认删除部门「${row.name}」吗？删除后不可恢复。`,
      "警告",
      {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning",
      }
    )
      .then(() => {
        DeptAPI.deleteByIds(String(row.id)).then(() => {
          ElMessage.success("删除成功");
          resetQuery();
        });
      })
      .catch(() => {});
  } else if (ids.value.length > 0) {
    const deptIds = ids.value;
    ElMessageBox.confirm(`确认删除选中的部门吗？删除后不可恢复。`, "警告", {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    })
      .then(() => {
        DeptAPI.deleteByIds(deptIds.join(",")).then(() => {
          ElMessage.success("删除成功");
          resetQuery();
        });
      })
      .catch(() => {});
  } else {
    ElMessage.warning("请勾选删除项");
  }
}

/** 关闭弹窗 */
function closeDialog() {
  dialog.visible = false;
  resetForm();
}

/** 重置表单  */
function resetForm() {
  deptFormRef.value.resetFields();
  deptFormRef.value.clearValidate();

  formData.id = undefined;
  formData.parentId = 0;
  formData.status = 1;
  formData.sort = 1;
}

onMounted(() => {
  handleQuery();
});
</script>
