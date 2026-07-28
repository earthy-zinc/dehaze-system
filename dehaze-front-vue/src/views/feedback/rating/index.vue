<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="用户名/评价内容"
            @keyup.enter="handleQuery"
          />
        </el-form-item>

        <el-form-item label="算法" prop="algorithmId">
          <el-select
            v-model="queryParams.algorithmId"
            clearable
            placeholder="全部"
            style="width: 160px"
          >
            <el-option
              v-for="item in algorithmOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="评分区间">
          <el-select
            v-model="queryParams.ratingMin"
            clearable
            placeholder="最低"
            style="width: 90px"
          >
            <el-option v-for="n in 5" :key="n" :label="n + '星'" :value="n" />
          </el-select>
          <span style="margin: 0 6px">-</span>
          <el-select
            v-model="queryParams.ratingMax"
            clearable
            placeholder="最高"
            style="width: 90px"
          >
            <el-option v-for="n in 5" :key="n" :label="n + '星'" :value="n" />
          </el-select>
        </el-form-item>

        <el-form-item label="有无评论" prop="hasComment">
          <el-select
            v-model="queryParams.hasComment"
            clearable
            placeholder="全部"
            style="width: 120px"
          >
            <el-option :value="true" label="有评论" />
            <el-option :value="false" label="无评论" />
          </el-select>
        </el-form-item>

        <el-form-item label="时间范围">
          <el-date-picker
            v-model="timeRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            value-format="YYYY-MM-DD"
            style="width: 260px"
            @change="handleTimeRangeChange"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleQuery"
            ><el-icon><Search /></el-icon>搜索</el-button
          >
          <el-button @click="resetQuery"
            ><el-icon><Refresh /></el-icon>重置</el-button
          >
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <div class="flex justify-end items-center">
          <el-button link type="primary" @click="goStats">
            <el-icon><DataLine /></el-icon>统计
          </el-button>
        </div>
      </template>

      <el-table
        ref="dataTableRef"
        v-loading="loading"
        :data="pageData"
        border
        highlight-current-row
        @selection-change="handleSelectionChange"
      >
        <el-table-column align="center" type="selection" width="55" />

        <el-table-column label="用户名" width="120">
          <template #default="scope">
            <span
              v-if="(scope.row as RatingPageVO).isAnonymous === 1"
              style="color: #909399"
              >匿名用户</span
            >
            <span v-else>{{
              (scope.row as RatingPageVO).username || "-"
            }}</span>
          </template>
        </el-table-column>

        <el-table-column label="算法" prop="algorithmName" width="120" />

        <el-table-column label="评分" width="120" align="center">
          <template #default="scope">
            <el-rate
              :model-value="(scope.row as RatingPageVO).rating"
              disabled
            />
          </template>
        </el-table-column>

        <el-table-column label="评价内容" min-width="200" show-overflow-tooltip>
          <template #default="scope">
            <span>{{ (scope.row as RatingPageVO).comment || "-" }}</span>
          </template>
        </el-table-column>

        <el-table-column label="标签" width="200">
          <template #default="scope">
            <template v-if="(scope.row as RatingPageVO).tags?.length">
              <el-tag
                v-for="(tag, idx) in (scope.row as RatingPageVO).tags!.slice(
                  0,
                  3
                )"
                :key="idx"
                size="small"
                style="margin: 2px"
                >{{ tag }}</el-tag
              >
              <el-tag
                v-if="(scope.row as RatingPageVO).tags!.length > 3"
                size="small"
                type="info"
                style="margin: 2px"
                >+{{ (scope.row as RatingPageVO).tags!.length - 3 }}</el-tag
              >
            </template>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <el-table-column label="评价时间" prop="createTime" width="170" />

        <el-table-column fixed="right" label="操作" width="200" align="center">
          <template #default="scope">
            <el-button
              link
              size="small"
              @click="handleDetail(scope.row as RatingPageVO)"
            >
              <el-icon><View /></el-icon>详情
            </el-button>
            <el-button
              v-hasPerm="['feedback:rating:edit']"
              link
              size="small"
              type="danger"
              @click="handleHide(scope.row as RatingPageVO)"
            >
              <el-icon><Hide /></el-icon
              >{{
                (scope.row as RatingPageVO).isHidden === 1 ? "显示" : "隐藏"
              }}
            </el-button>
            <el-button
              v-hasPerm="['feedback:rating:reply']"
              link
              size="small"
              type="primary"
              @click="openReplyDialog(scope.row as RatingPageVO)"
            >
              <el-icon><ChatLineRound /></el-icon>回复
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

    <el-dialog v-model="detailDialog.visible" title="评价详情" width="700px">
      <el-descriptions v-if="detailData" :column="2" border>
        <el-descriptions-item label="用户名">{{
          detailData.isAnonymous === 1 ? "匿名用户" : detailData.username || "-"
        }}</el-descriptions-item>
        <el-descriptions-item label="算法">{{
          detailData.algorithmName
        }}</el-descriptions-item>
        <el-descriptions-item label="评分">
          <el-rate :model-value="detailData.rating" disabled />
        </el-descriptions-item>
        <el-descriptions-item label="评价时间">{{
          detailData.createTime
        }}</el-descriptions-item>
        <el-descriptions-item label="标签">
          <template v-if="detailData.tags?.length">
            <el-tag
              v-for="(tag, idx) in detailData.tags"
              :key="idx"
              size="small"
              style="margin: 2px"
              >{{ tag }}</el-tag
            >
          </template>
          <span v-else>-</span>
        </el-descriptions-item>
        <el-descriptions-item label="是否匿名">{{
          detailData.isAnonymous === 1 ? "是" : "否"
        }}</el-descriptions-item>
        <el-descriptions-item label="是否隐藏">{{
          detailData.isHidden === 1 ? "是" : "否"
        }}</el-descriptions-item>
      </el-descriptions>

      <el-card v-if="detailData" shadow="never" style="margin-top: 12px">
        <template #header>评价内容</template>
        <div style="white-space: pre-wrap">
          {{ detailData.comment || "无" }}
        </div>
      </el-card>

      <div v-if="detailData?.imageUrls?.length" style="margin-top: 12px">
        <div style="margin-bottom: 8px">图片预览</div>
        <el-image
          v-for="(url, idx) in detailData.imageUrls"
          :key="idx"
          :src="url"
          :preview-src-list="detailData.imageUrls"
          :initial-index="idx"
          style="width: 100px; height: 100px; margin-right: 8px"
          fit="cover"
        />
      </div>

      <el-card
        v-if="detailData?.adminReply"
        shadow="never"
        style="margin-top: 12px"
      >
        <template #header>管理员回复</template>
        <div style="white-space: pre-wrap">{{ detailData.adminReply }}</div>
        <div
          v-if="detailData.replyTime"
          style="margin-top: 8px; color: #909399"
        >
          回复时间：{{ detailData.replyTime }}
        </div>
      </el-card>
    </el-dialog>

    <el-dialog
      v-model="replyDialog.visible"
      title="回复评价"
      width="600px"
      @close="closeReplyDialog"
    >
      <el-form
        ref="replyFormRef"
        :model="replyForm"
        :rules="replyRules"
        label-width="80px"
      >
        <el-form-item label="回复内容" prop="content">
          <el-input
            v-model="replyForm.content"
            type="textarea"
            :rows="4"
            placeholder="请输入回复内容（10-2000 字符）"
            maxlength="2000"
            show-word-limit
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleReplySubmit">确 定</el-button>
          <el-button @click="closeReplyDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  FeedbackAPI,
  AlgorithmAPI,
  RatingQuery,
  RatingPageVO,
  RatingDetailVO,
} from "dehaze-sdk-js";
import {
  Search,
  Refresh,
  View,
  Hide,
  ChatLineRound,
  DataLine,
} from "@element-plus/icons-vue";

defineOptions({ name: "FeedbackRating" });

const router = useRouter();
const queryFormRef = ref(ElForm);
const replyFormRef = ref(ElForm);
const loading = ref(false);
const pageData = ref<RatingPageVO[]>([]);
const total = ref(0);
const timeRange = ref<[string, string] | null>(null);
const selectedIds = ref<number[]>([]);

const queryParams = reactive<RatingQuery>({
  pageNum: 1,
  pageSize: 10,
});

const algorithmOptions = ref<{ value: number; label: string }[]>([]);

onMounted(() => {
  AlgorithmAPI.listAll().then((list) => {
    algorithmOptions.value = list.map((a) => ({ value: a.id, label: a.name }));
  });
});

function handleTimeRangeChange(val: [string, string] | null) {
  if (val && val.length === 2) {
    queryParams.startTime = `${val[0]} 00:00:00`;
    queryParams.endTime = `${val[1]} 23:59:59`;
  } else {
    queryParams.startTime = undefined;
    queryParams.endTime = undefined;
  }
}

function handleSelectionChange(rows: RatingPageVO[]) {
  selectedIds.value = rows.map((r) => r.id);
}

function handleQuery() {
  loading.value = true;
  FeedbackAPI.listRatings(queryParams)
    .then((data) => {
      pageData.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value?.resetFields();
  timeRange.value = null;
  queryParams.keywords = undefined;
  queryParams.algorithmId = undefined;
  queryParams.ratingMin = undefined;
  queryParams.ratingMax = undefined;
  queryParams.hasComment = undefined;
  queryParams.startTime = undefined;
  queryParams.endTime = undefined;
  queryParams.pageNum = 1;
  handleQuery();
}

function goStats() {
  router.push("/feedback/stats?tab=rating");
}

const detailDialog = reactive({ visible: false });
const detailData = ref<RatingDetailVO | null>(null);

function handleDetail(row: RatingPageVO) {
  detailData.value = row as RatingDetailVO;
  detailDialog.visible = true;
}

function handleHide(row: RatingPageVO) {
  const action = row.isHidden === 1 ? "显示" : "隐藏";
  ElMessageBox.confirm(`确认${action}该条评价吗？`, "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() => FeedbackAPI.hideRating(row.id))
    .then(() => {
      ElMessage.success(`${action}成功`);
      handleQuery();
    })
    .catch(() => {});
}

const replyDialog = reactive<{
  visible: boolean;
  loading: boolean;
  row: RatingPageVO | null;
}>({
  visible: false,
  loading: false,
  row: null,
});

const replyForm = reactive({ content: "" });

const replyRules = {
  content: [
    { required: true, message: "请输入回复内容", trigger: "blur" },
    {
      min: 10,
      max: 2000,
      message: "回复内容长度为 10-2000 字符",
      trigger: "blur",
    },
  ],
};

function openReplyDialog(row: RatingPageVO) {
  replyDialog.row = row;
  replyForm.content = "";
  replyDialog.visible = true;
}

function closeReplyDialog() {
  replyDialog.visible = false;
  replyDialog.row = null;
  replyForm.content = "";
  replyFormRef.value?.resetFields();
}

function handleReplySubmit() {
  if (!replyDialog.row) return;
  replyFormRef.value?.validate((valid: boolean) => {
    if (!valid) return;
    replyDialog.loading = true;
    FeedbackAPI.replyRating(replyDialog.row!.id, replyForm.content)
      .then(() => {
        ElMessage.success("回复成功");
        closeReplyDialog();
        handleQuery();
      })
      .finally(() => {
        replyDialog.loading = false;
      });
  });
}

onMounted(() => {
  handleQuery();
});
</script>
