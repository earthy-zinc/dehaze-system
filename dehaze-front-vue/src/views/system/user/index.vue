<!-- 用户管理 -->
<template>
  <div class="app-container">
    <el-row :gutter="20">
      <!-- 部门树 -->
      <el-col :lg="4" :xs="24" class="mb-[12px]">
        <dept-tree v-model="queryParams.deptId" @node-click="handleQuery" />
      </el-col>

      <!-- 用户列表 -->
      <el-col :lg="20" :xs="24">
        <div class="search-container">
          <el-form ref="queryFormRef" :inline="true" :model="queryParams">
            <el-form-item label="关键字" prop="keywords">
              <el-input
                v-model="queryParams.keywords"
                clearable
                placeholder="用户名/昵称/手机号"
                style="width: 200px"
                @keyup.enter="handleQuery"
                @input="debouncedQuery"
              />
            </el-form-item>

            <el-form-item label="状态" prop="status">
              <el-select
                v-model="queryParams.status"
                class="!w-[100px]"
                clearable
                placeholder="全部"
              >
                <el-option label="启用" value="1" />
                <el-option label="禁用" value="0" />
              </el-select>
            </el-form-item>

            <el-form-item label="创建时间">
              <el-date-picker
                v-model="dateTimeRange"
                class="!w-[240px]"
                end-placeholder="截止时间"
                range-separator="~"
                start-placeholder="开始时间"
                type="daterange"
                value-format="YYYY-MM-DD"
              />
            </el-form-item>

            <el-form-item>
              <el-button type="primary" @click="handleQuery"
                ><el-icon><Search /></el-icon>搜索</el-button
              >
              <el-button @click="resetQuery">
                <el-icon><Refresh /></el-icon>
                重置</el-button
              >
            </el-form-item>
          </el-form>
        </div>

        <el-card class="table-container" shadow="never">
          <template #header>
            <div class="flex justify-between">
              <div>
                <el-button
                  v-hasPerm="['sys:user:add']"
                  type="success"
                  @click="openDialog('user-form')"
                  ><el-icon><Plus /></el-icon>新增</el-button
                >
                <el-button
                  v-hasPerm="['sys:user:delete']"
                  :disabled="removeIds.length === 0"
                  type="danger"
                  @click="handleDelete()"
                  ><el-icon><Delete /></el-icon>删除</el-button
                >
              </div>
              <div>
                <ImportExportToolbar
                  module="user"
                  :query-params="queryParams"
                  :extra-import-params="{ deptId: queryParams.deptId }"
                  @import-complete="handleQuery"
                />
              </div>
            </div>
          </template>

          <el-table
            v-loading="loading"
            :data="pageData"
            @selection-change="handleSelectionChange"
          >
            <el-table-column align="center" type="selection" width="50" />
            <el-table-column
              key="id"
              align="center"
              label="编号"
              prop="id"
              width="100"
            />
            <el-table-column
              key="username"
              align="center"
              label="用户名"
              prop="username"
            />
            <el-table-column
              align="center"
              label="用户昵称"
              prop="nickname"
              width="120"
            />

            <el-table-column
              align="center"
              label="性别"
              prop="genderLabel"
              width="100"
            />

            <el-table-column
              align="center"
              label="部门"
              prop="deptName"
              width="120"
            />
            <el-table-column
              align="center"
              label="手机号码"
              prop="mobile"
              width="120"
            />

            <el-table-column align="center" label="状态" prop="status">
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
              label="创建时间"
              prop="createTime"
              width="180"
            />
            <el-table-column fixed="right" label="操作" width="220">
              <template #default="scope">
                <el-button
                  v-hasPerm="['sys:user:password:reset']"
                  link
                  size="small"
                  type="primary"
                  @click="resetPassword(scope.row)"
                  ><el-icon><RefreshLeft /></el-icon>重置密码</el-button
                >
                <el-button
                  v-hasPerm="['sys:user:edit']"
                  link
                  size="small"
                  type="primary"
                  @click="openDialog('user-form', scope.row.id)"
                  ><el-icon><Edit /></el-icon>编辑</el-button
                >
                <el-button
                  v-hasPerm="['sys:user:delete']"
                  link
                  size="small"
                  type="primary"
                  @click="handleDelete(scope.row)"
                  ><el-icon><Delete /></el-icon>删除</el-button
                >
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
      </el-col>
    </el-row>

    <!-- 弹窗 -->
    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      :width="dialog.width"
      append-to-body
      @close="closeDialog"
    >
      <!-- 用户新增/编辑表单 -->
      <el-form
        v-if="dialog.type === 'user-form'"
        ref="userFormRef"
        :model="formData"
        :rules="rules"
        label-width="80px"
      >
        <el-form-item label="用户名" prop="username">
          <el-input
            v-model="formData.username"
            :readonly="!!formData.id"
            placeholder="请输入用户名"
          />
        </el-form-item>

        <el-form-item label="用户昵称" prop="nickname">
          <el-input v-model="formData.nickname" placeholder="请输入用户昵称" />
        </el-form-item>

        <el-form-item label="所属部门" prop="deptId">
          <el-tree-select
            v-model="formData.deptId"
            :data="deptList"
            :render-after-expand="false"
            check-strictly
            filterable
            placeholder="请选择所属部门"
          />
        </el-form-item>

        <el-form-item label="性别" prop="gender">
          <dictionary v-model="formData.gender" type-code="gender" />
        </el-form-item>

        <el-form-item label="角色" prop="roleIds">
          <el-select v-model="formData.roleIds" multiple placeholder="请选择">
            <el-option
              v-for="item in roleList"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="手机号码" prop="mobile">
          <el-input
            v-model="formData.mobile"
            maxlength="11"
            placeholder="请输入手机号码"
          />
        </el-form-item>

        <el-form-item label="邮箱" prop="email">
          <el-input
            v-model="formData.email"
            maxlength="50"
            placeholder="请输入邮箱"
          />
        </el-form-item>

        <el-form-item label="状态" prop="status">
          <el-radio-group v-model="formData.status">
            <el-radio :label="1">正常</el-radio>
            <el-radio :label="0">禁用</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>

      <!-- 弹窗底部操作按钮 -->
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleSubmit">确 定</el-button>
          <el-button @click="closeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
defineOptions({
  name: "SystemUser",
  inheritAttrs: false,
});

import {
  DeptAPI,
  OptionType,
  RoleAPI,
  UserAPI,
  UserForm,
  UserPageVO,
  UserQuery,
} from "dehaze-sdk-js";

import {
  Delete,
  Edit,
  Plus,
  Refresh,
  RefreshLeft,
  Search,
} from "@element-plus/icons-vue";

import ImportExportToolbar from "@/components/ImportExportToolbar/index.vue";

const queryFormRef = ref(ElForm); // 查询表单
const userFormRef = ref(ElForm); // 用户表单

const loading = ref(false); //  加载状态
const removeIds = ref([]); // 删除用户ID集合 用于批量删除
const queryParams = reactive<UserQuery>({
  pageNum: 1,
  pageSize: 10,
});
const dateTimeRange = ref("");
const total = ref(0); // 数据总数
const pageData = ref<UserPageVO[]>(); // 用户分页数据
const deptList = ref<OptionType[]>(); // 部门下拉数据源
const roleList = ref<OptionType[]>(); // 角色下拉数据源

watch(dateTimeRange, (newVal) => {
  if (newVal) {
    queryParams.startTime = newVal[0];
    queryParams.endTime = newVal[1];
  } else {
    queryParams.startTime = undefined;
    queryParams.endTime = undefined;
  }
});

// 弹窗对象
const dialog = reactive({
  visible: false,
  type: "user-form",
  width: 800,
  title: "",
});

// 用户表单数据
const formData = reactive<UserForm>({
  status: 1,
});

// 校验规则
const rules = reactive({
  username: [{ required: true, message: "用户名不能为空", trigger: "blur" }],
  nickname: [{ required: true, message: "用户昵称不能为空", trigger: "blur" }],
  deptId: [{ required: true, message: "所属部门不能为空", trigger: "blur" }],
  gender: [{ required: true, message: "性别不能为空", trigger: "change" }],
  roleIds: [{ required: true, message: "用户角色不能为空", trigger: "blur" }],
  email: [
    {
      pattern: /\w[-\w.+]*@([A-Za-z0-9][-A-Za-z0-9]+\.)+[A-Za-z]{2,14}/,
      message: "请输入正确的邮箱地址",
      trigger: "blur",
    },
  ],
  mobile: [
    {
      pattern: /^1[3|4|5|6|7|8|9][0-9]\d{8}$/,
      message: "请输入正确的手机号码",
      trigger: "blur",
    },
  ],
});

/** 查询 */
function handleQuery() {
  loading.value = true;
  UserAPI.getPage(queryParams)
    .then((data) => {
      pageData.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

/** 防抖查询（300ms） */
const debouncedQuery = useDebounceFn(handleQuery, 300);

/** 修改用户状态 */
function handleStatusChange(row: any) {
  UserAPI.updateStatus(row.id, row.status)
    .then(() => {
      ElMessage.success("状态修改成功");
    })
    .catch(() => {
      row.status = row.status === 1 ? 0 : 1;
    });
}

/** 重置查询 */
function resetQuery() {
  queryFormRef.value.resetFields();
  dateTimeRange.value = "";
  queryParams.pageNum = 1;
  queryParams.deptId = undefined;
  queryParams.startTime = undefined;
  queryParams.endTime = undefined;
  handleQuery();
}

/** 行选中 */
function handleSelectionChange(selection: any) {
  removeIds.value = selection.map((item: any) => item.id);
}

/** 重置密码 */
function resetPassword(row: { [key: string]: any }) {
  ElMessageBox.prompt(
    "请输入用户「" + row.username + "」的新密码",
    "重置密码",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
    }
  )
    .then(({ value }) => {
      if (!value) {
        ElMessage.warning("请输入新密码");
        return false;
      }
      UserAPI.updatePassword(row.id, value).then(() => {
        ElMessage.success("密码重置成功，新密码是：" + value);
      });
    })
    .catch(() => {});
}

/** 加载角色下拉数据源 */
async function loadRoleOptions() {
  roleList.value = await RoleAPI.getOptions();
}

/** 加载部门下拉数据源 */
async function loadDeptOptions() {
  deptList.value = await DeptAPI.getOptions();
}

/**
 * 打开弹窗
 *
 * @param type 弹窗类型  用户表单：user-form
 * @param id 用户ID
 */
async function openDialog(type: string, id?: number) {
  dialog.visible = true;
  dialog.type = type;

  if (dialog.type === "user-form") {
    // 用户表单弹窗
    await loadDeptOptions();
    await loadRoleOptions();
    if (id) {
      dialog.title = "修改用户";
      const data = await UserAPI.getFormData(id);
      Object.assign(formData, { ...data });
    } else {
      dialog.title = "新增用户";
    }
  }
}

/**
 * 关闭弹窗
 *
 * @param type 弹窗类型  用户表单：user-form
 */
function closeDialog() {
  dialog.visible = false;
  if (dialog.type === "user-form") {
    userFormRef.value.resetFields();
    userFormRef.value.clearValidate();

    formData.id = undefined;
    formData.status = 1;
  }
}

/** 表单提交 */
const handleSubmit = useThrottleFn(() => {
  if (dialog.type === "user-form") {
    userFormRef.value.validate((valid: any) => {
      if (valid) {
        const userId = formData.id;
        loading.value = true;
        if (userId) {
          UserAPI.update(userId, formData)
            .then(() => {
              ElMessage.success("修改用户成功");
              closeDialog();
              resetQuery();
            })
            .finally(() => (loading.value = false));
        } else {
          UserAPI.add(formData)
            .then(() => {
              ElMessage.success("新增用户成功");
              closeDialog();
              resetQuery();
            })
            .finally(() => (loading.value = false));
        }
      }
    });
  }
}, 3000);

/** 删除用户 */
function handleDelete(row?: any) {
  let userIds: string;
  let confirmText: string;

  if (row) {
    userIds = String(row.id);
    confirmText = `确认删除用户「${row.username}」吗？删除后不可恢复。`;
  } else {
    if (removeIds.value.length === 0) {
      ElMessage.warning("请勾选删除项");
      return;
    }
    userIds = removeIds.value.join(",");
    confirmText = `确认删除选中的 ${removeIds.value.length} 个用户吗？删除后不可恢复。`;
  }

  ElMessageBox.confirm(confirmText, "警告", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(function () {
      UserAPI.deleteByIds(userIds).then(() => {
        ElMessage.success("删除成功");
        resetQuery();
      });
    })
    .catch(() => {});
}

onMounted(async () => {
  // 默认选中部门树第一个根节点
  const deptOptions = await DeptAPI.getOptions();
  if (deptOptions && deptOptions.length > 0) {
    queryParams.deptId = Number(deptOptions[0].value);
  }
  handleQuery();
});
</script>
