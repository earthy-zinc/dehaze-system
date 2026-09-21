<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="用户名/昵称"
            style="width: 200px"
            @keyup.enter="handleQuery"
          />
        </el-form-item>

        <el-form-item label="等级" prop="levelCode">
          <el-select
            v-model="queryParams.levelCode"
            class="!w-[140px]"
            clearable
            placeholder="全部"
          >
            <el-option
              v-for="opt in levelOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="状态" prop="status">
          <el-select
            v-model="queryParams.status"
            class="!w-[100px]"
            clearable
            placeholder="全部"
          >
            <el-option label="正常" :value="1" />
            <el-option label="冻结" :value="0" />
          </el-select>
        </el-form-item>

        <el-form-item label="到期时间">
          <el-date-picker
            v-model="expireTimeRange"
            class="!w-[240px]"
            end-placeholder="截止时间"
            range-separator="~"
            start-placeholder="开始时间"
            type="daterange"
            value-format="YYYY-MM-DD"
          />
        </el-form-item>

        <el-form-item label="成长值">
          <el-input
            v-model.number="queryParams.growthMin"
            class="!w-[100px]"
            placeholder="最小"
            type="number"
          />
          <span class="mx-1">~</span>
          <el-input
            v-model.number="queryParams.growthMax"
            class="!w-[100px]"
            placeholder="最大"
            type="number"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleQuery">
            <el-icon><Search /></el-icon>搜索
          </el-button>
          <el-button @click="resetQuery">
            <el-icon><Refresh /></el-icon>重置
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <el-button
            v-hasPerm="['member:benefit:edit']"
            type="warning"
            @click="openBenefitDrawer"
          >
            <el-icon><Star /></el-icon>权益配置
          </el-button>
          <el-button @click="handleExport">
            <el-icon><Download /></el-icon>导出
          </el-button>
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="pageData"
        border
        highlight-current-row
        @selection-change="handleSelectionChange"
      >
        <el-table-column align="center" type="selection" width="55" />
        <el-table-column label="用户名" prop="username" min-width="120" />
        <el-table-column label="昵称" prop="nickname" min-width="120" />
        <el-table-column align="center" label="等级" width="110">
          <template #default="scope">
            <el-tag
              :color="levelColorMap[scope.row.levelCode as MemberLevelCode]"
              disable-transitions
              effect="dark"
            >
              {{ scope.row.levelName }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          align="center"
          label="成长值"
          prop="growthValue"
          width="100"
        />
        <el-table-column
          align="center"
          label="本月已用"
          prop="monthlyUsed"
          width="100"
        />
        <el-table-column align="center" label="到期时间" width="180">
          <template #default="scope">
            <span v-if="scope.row.expireTime">{{ scope.row.expireTime }}</span>
            <el-tag v-else disable-transitions effect="plain" type="info">
              成长值维持
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column align="center" label="状态" width="90">
          <template #default="scope">
            <el-tag
              :type="scope.row.status === 1 ? 'success' : 'danger'"
              disable-transitions
              effect="light"
            >
              {{ scope.row.status === 1 ? "正常" : "冻结" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          align="center"
          label="开通时间"
          prop="becomeMemberTime"
          width="180"
        />
        <el-table-column fixed="right" label="操作" width="320">
          <template #default="scope">
            <el-button
              link
              size="small"
              type="primary"
              @click="openDetailDialog(scope.row as MemberPageVO)"
            >
              <el-icon><View /></el-icon>详情
            </el-button>
            <el-button
              v-hasPerm="['member:level:edit']"
              link
              size="small"
              type="primary"
              @click="openLevelDialog(scope.row as MemberPageVO)"
            >
              <el-icon><Edit /></el-icon>等级
            </el-button>
            <el-button
              v-hasPerm="['member:growth:edit']"
              link
              size="small"
              type="primary"
              @click="openGrowthDialog(scope.row as MemberPageVO)"
            >
              <el-icon><ArrowUp /></el-icon>成长值
            </el-button>
            <el-button
              v-hasPerm="['member:status:edit']"
              link
              size="small"
              :type="scope.row.status === 1 ? 'danger' : 'success'"
              @click="openFreezeDialog(scope.row as MemberPageVO)"
            >
              <el-icon>
                <component :is="scope.row.status === 1 ? Lock : Unlock" />
              </el-icon>
              {{ scope.row.status === 1 ? "冻结" : "解冻" }}
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

    <!-- 会员详情弹窗 -->
    <el-dialog
      v-model="detailDialog.visible"
      title="会员详情"
      width="820px"
      @close="closeDetailDialog"
    >
      <el-tabs
        v-loading="detailDialog.loading"
        v-model="detailDialog.activeTab"
        @tab-change="handleDetailTabChange"
      >
        <el-tab-pane label="基本信息" name="basic">
          <el-descriptions v-if="detailData" :column="2" border>
            <el-descriptions-item label="用户名">{{
              detailData.username
            }}</el-descriptions-item>
            <el-descriptions-item label="昵称">{{
              detailData.nickname
            }}</el-descriptions-item>
            <el-descriptions-item label="等级">
              <el-tag
                :color="levelColorMap[detailData.levelCode]"
                disable-transitions
                effect="dark"
              >
                {{ detailData.levelName }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="成长值">{{
              detailData.growthValue
            }}</el-descriptions-item>
            <el-descriptions-item label="到期时间">
              {{ detailData.expireTime || "成长值维持" }}
            </el-descriptions-item>
            <el-descriptions-item label="状态">
              <el-tag
                :type="detailData.status === 1 ? 'success' : 'danger'"
                effect="light"
              >
                {{ detailData.status === 1 ? "正常" : "冻结" }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="开通时间">{{
              detailData.becomeMemberTime || "-"
            }}</el-descriptions-item>
            <el-descriptions-item label="累计消费">{{
              detailData.totalConsumption
            }}</el-descriptions-item>
            <el-descriptions-item label="本月已用次数">{{
              detailData.monthlyUsed
            }}</el-descriptions-item>
            <el-descriptions-item label="月度去雾配额">{{
              detailData.benefits?.monthlyDehazeQuota
            }}</el-descriptions-item>
            <el-descriptions-item label="月度评估配额">{{
              detailData.benefits?.monthlyEvaluateQuota
            }}</el-descriptions-item>
            <el-descriptions-item
              v-if="detailData.frozenReason"
              label="冻结原因"
              :span="2"
            >
              {{ detailData.frozenReason }}（{{ detailData.frozenTime }}）
            </el-descriptions-item>
          </el-descriptions>
        </el-tab-pane>

        <el-tab-pane label="成长值流水" name="growthLogs" lazy>
          <el-table
            v-loading="growthLogs.loading"
            :data="growthLogs.list"
            border
            size="small"
          >
            <el-table-column
              align="center"
              label="时间"
              prop="createTime"
              width="170"
            />
            <el-table-column align="center" label="类型" width="130">
              <template #default="scope">
                <el-tag
                  :type="scope.row.changeValue >= 0 ? 'success' : 'danger'"
                  disable-transitions
                  effect="light"
                >
                  {{
                    changeTypeLabels[
                      scope.row.changeType as GrowthChangeType
                    ] ?? scope.row.changeType
                  }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column
              align="center"
              label="变动值"
              prop="changeValue"
              width="90"
            />
            <el-table-column
              align="center"
              label="变动后余额"
              prop="balance"
              width="110"
            />
            <el-table-column label="关联业务" min-width="160">
              <template #default="scope">
                {{ scope.row.relatedId || "-" }}
              </template>
            </el-table-column>
          </el-table>
          <pagination
            v-if="growthLogs.total > 0"
            v-model:limit="growthLogs.pageSize"
            v-model:page="growthLogs.pageNum"
            v-model:total="growthLogs.total"
            layout="total, prev, pager, next"
            @pagination="loadGrowthLogs"
          />
        </el-tab-pane>

        <el-tab-pane label="消费记录" name="consumption" lazy>
          <el-table
            v-loading="consumption.loading"
            :data="consumption.list"
            border
            size="small"
          >
            <el-table-column label="订单号" min-width="180">
              <template #default="scope">
                {{ scope.row.orderNo }}
              </template>
            </el-table-column>
            <el-table-column
              label="商品名称"
              prop="packageName"
              min-width="140"
            />
            <el-table-column
              align="center"
              label="金额"
              prop="payableAmount"
              width="100"
            />
            <el-table-column align="center" label="支付时间" width="170">
              <template #default="scope">
                {{ scope.row.paidTime || scope.row.createTime }}
              </template>
            </el-table-column>
            <el-table-column align="center" label="状态" width="100">
              <template #default="scope">
                <el-tag disable-transitions effect="light">
                  {{
                    orderStatusLabels[scope.row.status as OrderStatus] ??
                    scope.row.status
                  }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
          <pagination
            v-if="consumption.total > 0"
            v-model:limit="consumption.pageSize"
            v-model:page="consumption.pageNum"
            v-model:total="consumption.total"
            layout="total, prev, pager, next"
            @pagination="loadConsumptionRecords"
          />
        </el-tab-pane>

        <el-tab-pane label="权益使用" name="benefitUsage" lazy>
          <div v-loading="benefitUsage.loading">
            <template v-if="benefitUsage.data">
              <el-table :data="benefitUsageRows" border size="small">
                <el-table-column
                  label="服务项目"
                  prop="label"
                  min-width="140"
                />
                <el-table-column
                  align="center"
                  label="配额"
                  prop="quota"
                  width="100"
                />
                <el-table-column
                  align="center"
                  label="已用"
                  prop="used"
                  width="100"
                />
                <el-table-column
                  align="center"
                  label="剩余"
                  prop="remaining"
                  width="100"
                />
              </el-table>
              <el-descriptions :column="2" border class="mt-4">
                <el-descriptions-item label="AI 积分余额">
                  {{ benefitUsage.data.aiCategory.creditsBalance }}
                </el-descriptions-item>
                <el-descriptions-item label="今日已用">
                  {{ benefitUsage.data.aiCategory.todayUsed }}
                </el-descriptions-item>
                <el-descriptions-item label="AI 日限额">
                  {{ benefitUsage.data.aiCategory.dailyLimit }}
                </el-descriptions-item>
                <el-descriptions-item label="AI 月限额">
                  {{ benefitUsage.data.aiCategory.monthlyLimit }}
                </el-descriptions-item>
              </el-descriptions>
            </template>
            <el-empty
              v-else-if="!benefitUsage.loading"
              description="暂无数据"
              :image-size="80"
            />
          </div>
        </el-tab-pane>

        <el-tab-pane label="操作日志" name="operationLogs" lazy>
          <el-table
            v-loading="operationLogs.loading"
            :data="operationLogs.list"
            border
            size="small"
          >
            <el-table-column
              align="center"
              label="时间"
              prop="createTime"
              width="170"
            />
            <el-table-column align="center" label="操作类型" width="120">
              <template #default="scope">
                {{
                  auditActionLabels[scope.row.action as MemberAuditAction] ??
                  scope.row.action
                }}
              </template>
            </el-table-column>
            <el-table-column label="变更内容" min-width="240">
              <template #default="scope">
                {{ formatAuditChange(scope.row) }}
              </template>
            </el-table-column>
            <el-table-column
              align="center"
              label="操作人ID"
              prop="operatorId"
              width="100"
            />
          </el-table>
          <pagination
            v-if="operationLogs.total > 0"
            v-model:limit="operationLogs.pageSize"
            v-model:page="operationLogs.pageNum"
            v-model:total="operationLogs.total"
            layout="total, prev, pager, next"
            @pagination="loadOperationLogs"
          />
        </el-tab-pane>
      </el-tabs>
    </el-dialog>

    <!-- 等级调整弹窗 -->
    <el-dialog
      v-model="levelDialog.visible"
      title="等级调整"
      width="500px"
      @close="resetLevelForm"
    >
      <el-form
        ref="levelFormRef"
        :model="levelForm"
        :rules="levelRules"
        label-width="100px"
      >
        <el-form-item label="会员">
          <span
            >{{ levelDialog.username }}（当前：{{
              levelDialog.currentLevelName
            }}）</span
          >
        </el-form-item>
        <el-form-item label="目标等级" prop="levelCode">
          <el-select
            v-model="levelForm.levelCode"
            placeholder="请选择等级"
            style="width: 100%"
          >
            <el-option
              v-for="opt in levelOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="到期时间" prop="expireTime">
          <el-date-picker
            v-model="levelForm.expireTime"
            placeholder="不选则由成长值维持"
            style="width: 100%"
            type="date"
            value-format="YYYY-MM-DD HH:mm:ss"
          />
        </el-form-item>
        <el-form-item label="调整原因" prop="reason">
          <el-input
            v-model="levelForm.reason"
            :rows="3"
            :maxlength="200"
            show-word-limit
            type="textarea"
            placeholder="请输入2-200字符的调整原因"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button type="primary" @click="submitLevelAdjust">确 定</el-button>
        <el-button @click="levelDialog.visible = false">取 消</el-button>
      </template>
    </el-dialog>

    <!-- 成长值调整弹窗 -->
    <el-dialog
      v-model="growthDialog.visible"
      title="成长值调整"
      width="500px"
      @close="resetGrowthForm"
    >
      <el-form
        ref="growthFormRef"
        :model="growthForm"
        :rules="growthRules"
        label-width="100px"
      >
        <el-form-item label="会员">
          <span>{{ growthDialog.username }}</span>
        </el-form-item>
        <el-form-item label="当前成长值">
          <span>{{ growthDialog.currentGrowth }}</span>
        </el-form-item>
        <el-form-item label="变动值" prop="changeValue">
          <el-input-number
            v-model="growthForm.changeValue"
            :precision="0"
            :step="1"
            controls-position="right"
            style="width: 200px"
          />
          <span class="ml-2 text-secondary">正数为增加，负数为扣减</span>
        </el-form-item>
        <el-form-item label="预览">
          <span>
            {{ growthDialog.currentGrowth }} + {{ growthForm.changeValue }} =
            <strong>{{ expectedGrowth }}</strong>
          </span>
        </el-form-item>
        <el-form-item label="调整原因" prop="reason">
          <el-input
            v-model="growthForm.reason"
            :rows="3"
            :maxlength="200"
            show-word-limit
            type="textarea"
            placeholder="请输入调整原因"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button type="primary" @click="submitGrowthAdjust">确 定</el-button>
        <el-button @click="growthDialog.visible = false">取 消</el-button>
      </template>
    </el-dialog>

    <!-- 冻结/解冻弹窗 -->
    <el-dialog
      v-model="freezeDialog.visible"
      :title="freezeForm.status === 0 ? '冻结会员' : '解冻会员'"
      width="500px"
      @close="resetFreezeForm"
    >
      <el-form
        ref="freezeFormRef"
        :model="freezeForm"
        :rules="freezeRules"
        label-width="100px"
      >
        <el-form-item label="会员">
          <span>{{ freezeDialog.username }}</span>
        </el-form-item>
        <el-form-item
          v-if="freezeForm.status === 0"
          label="冻结原因"
          prop="reason"
        >
          <el-input
            v-model="freezeForm.reason"
            :rows="3"
            :maxlength="200"
            show-word-limit
            type="textarea"
            placeholder="请输入冻结原因"
          />
        </el-form-item>
        <el-form-item v-else label="说明">
          <span>解冻后会员可正常使用所有权益</span>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button type="primary" @click="submitStatusChange">确 定</el-button>
        <el-button @click="freezeDialog.visible = false">取 消</el-button>
      </template>
    </el-dialog>

    <!-- 权益配置抽屉 -->
    <el-drawer
      v-model="benefitDrawer.visible"
      title="权益配置"
      size="60%"
      @open="loadBenefits"
    >
      <el-table v-loading="benefitDrawer.loading" :data="benefitList" border>
        <el-table-column align="center" label="等级" width="110">
          <template #default="scope">
            <el-tag
              :color="levelColorMap[scope.row.levelCode as MemberLevelCode]"
              disable-transitions
              effect="dark"
            >
              {{ scope.row.levelName }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column align="center" label="月去雾配额" width="120">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.monthlyDehazeQuota"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 100px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="月去雨配额" width="120">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.monthlyDerainQuota"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 100px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="月去雪配额" width="120">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.monthlyDesnowQuota"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 100px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="月低光配额" width="120">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.monthlyLowlightQuota"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 100px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="月超分配额" width="120">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.monthlySuperResolutionQuota"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 100px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="月去噪配额" width="120">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.monthlyDenoiseQuota"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 100px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="月修复配额" width="120">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.monthlyInpaintQuota"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 100px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="月评估配额" width="120">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.monthlyEvaluateQuota"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 100px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="AI日限额" width="130">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.aiCreditsDaily"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 110px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="AI月限额" width="140">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.aiCreditsMonthly"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 120px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="多模态日限额" width="130">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.multimodalLimit"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 110px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="在线设备数" width="130">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.maxDevices"
              :min="1"
              controls-position="right"
              size="small"
              style="width: 110px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="VIP月赠积分" width="130">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.vipGiftCredits"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 110px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="历史保留" width="110">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.historyRetention"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 90px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="批量上限" width="110">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.batchLimit"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 90px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="优先级" width="100">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.priority"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 80px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="高级参数" width="110">
          <template #default="scope">
            <el-input-number
              v-model="scope.row.advancedParams"
              :min="0"
              controls-position="right"
              size="small"
              style="width: 90px"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="高清导出" width="90">
          <template #default="scope">
            <el-switch
              v-model="scope.row.hdExport"
              :active-value="1"
              :inactive-value="0"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="报告导出" width="90">
          <template #default="scope">
            <el-switch
              v-model="scope.row.reportExport"
              :active-value="1"
              :inactive-value="0"
            />
          </template>
        </el-table-column>
        <el-table-column align="center" label="批量下载" width="90">
          <template #default="scope">
            <el-switch
              v-model="scope.row.batchDownload"
              :active-value="1"
              :inactive-value="0"
            />
          </template>
        </el-table-column>
        <el-table-column fixed="right" label="操作" width="80">
          <template #default="scope">
            <el-button
              link
              size="small"
              type="primary"
              @click="saveBenefit(scope.row as BenefitVO)"
              >保存</el-button
            >
          </template>
        </el-table-column>
      </el-table>
    </el-drawer>
  </div>
</template>

<script lang="ts" setup>
import {
  MemberAPI,
  MemberQuery,
  MemberPageVO,
  MemberDetailVO,
  MemberLevelAdjustForm,
  MemberGrowthAdjustForm,
  MemberStatusForm,
  BenefitForm,
  BenefitVO,
  BenefitSummaryVO,
  BenefitTaskType,
  GrowthChangeType,
  GrowthLogVO,
  MemberAuditLogVO,
  MemberAuditAction,
  MemberLevelCode,
  MyOrderVO,
  OrderStatus,
} from "dehaze-sdk-js";
import {
  Search,
  Refresh,
  Download,
  Edit,
  View,
  Lock,
  Unlock,
  ArrowUp,
  Star,
} from "@element-plus/icons-vue";

defineOptions({
  name: "MemberList",
  inheritAttrs: false,
});

const queryFormRef = ref(ElForm);
const levelFormRef = ref(ElForm);
const growthFormRef = ref(ElForm);
const freezeFormRef = ref(ElForm);

const loading = ref(false);
const total = ref(0);
const pageData = ref<MemberPageVO[]>([]);
const ids = ref<number[]>([]);
const expireTimeRange = ref<[string, string] | null>(null);

const queryParams = reactive<MemberQuery>({
  pageNum: 1,
  pageSize: 10,
});

const levelColorMap: Record<MemberLevelCode, string> = {
  level_0: "#8c8c8c",
  level_1: "#409eff",
  level_2: "#722ed1",
  level_3: "#fa8c16",
};

const levelOrderMap: Record<MemberLevelCode, number> = {
  level_0: 0,
  level_1: 1,
  level_2: 2,
  level_3: 3,
};

const levelOptions: { label: string; value: MemberLevelCode }[] = [
  { label: "普通用户", value: "level_0" },
  { label: "VIP1", value: "level_1" },
  { label: "VIP2", value: "level_2" },
  { label: "SVIP", value: "level_3" },
];

watch(expireTimeRange, (val) => {
  if (val && Array.isArray(val)) {
    queryParams.expireTimeStart = val[0];
    queryParams.expireTimeEnd = val[1];
  } else {
    queryParams.expireTimeStart = undefined;
    queryParams.expireTimeEnd = undefined;
  }
});

function handleQuery() {
  loading.value = true;
  MemberAPI.getPage(queryParams)
    .then((data) => {
      pageData.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value.resetFields();
  expireTimeRange.value = null;
  queryParams.pageNum = 1;
  queryParams.expireTimeStart = undefined;
  queryParams.expireTimeEnd = undefined;
  queryParams.growthMin = undefined;
  queryParams.growthMax = undefined;
  handleQuery();
}

function handleSelectionChange(selection: MemberPageVO[]) {
  ids.value = selection.map((item) => item.userId);
}

function handleExport() {
  ElMessage.warning("导出功能暂未上线，请稍后使用");
}

// ==================== 详情弹窗（5 页签按需加载） ====================
const detailDialog = reactive({
  visible: false,
  loading: false,
  userId: 0,
  activeTab: "basic",
});
const detailData = ref<MemberDetailVO>();

const growthLogs = reactive({
  loading: false,
  loaded: false,
  list: [] as GrowthLogVO[],
  total: 0,
  pageNum: 1,
  pageSize: 10,
});

const consumption = reactive({
  loading: false,
  loaded: false,
  list: [] as MyOrderVO[],
  total: 0,
  pageNum: 1,
  pageSize: 10,
});

const benefitUsage = reactive({
  loading: false,
  loaded: false,
  data: undefined as BenefitSummaryVO | undefined,
});

const operationLogs = reactive({
  loading: false,
  loaded: false,
  list: [] as MemberAuditLogVO[],
  total: 0,
  pageNum: 1,
  pageSize: 10,
});

const changeTypeLabels: Record<GrowthChangeType, string> = {
  process: "图像处理",
  evaluate: "效果评估",
  rating: "评价",
  sign_in: "签到",
  sign_in_bonus: "连续签到奖励",
  consume: "消费",
  refund_deduct: "退款扣减",
  admin_adjust: "管理员调整",
  ai_consume: "AI 对话",
};

const orderStatusLabels: Record<OrderStatus, string> = {
  pending: "待支付",
  paid: "已支付",
  completed: "已完成",
  cancelled: "已取消",
  refunding: "退款中",
  refunded: "已退款",
};

const auditActionLabels: Record<MemberAuditAction, string> = {
  level_change: "等级调整",
  growth_change: "成长值调整",
  status_change: "冻结/解冻",
};

function formatAuditChange(
  row: Pick<MemberAuditLogVO, "beforeValue" | "afterValue">
) {
  const before = row.beforeValue ? JSON.stringify(row.beforeValue) : "";
  const after = row.afterValue ? JSON.stringify(row.afterValue) : "";
  if (!before && !after) return "-";
  if (!before) return after;
  if (!after) return before;
  return `${before} → ${after}`;
}

const taskTypeLabels: Record<BenefitTaskType, string> = {
  dehaze: "去雾",
  derain: "去雨",
  desnow: "去雪",
  lowlight: "低光",
  super_resolution: "超分",
  denoise: "去噪",
  inpaint: "修复",
};

const benefitUsageRows = computed(() => {
  const data = benefitUsage.data;
  if (!data) return [];
  const rows: {
    label: string;
    quota: number;
    used: number;
    remaining: number;
  }[] = [];
  [
    ...(data.imageCategory.details ?? []),
    ...(data.evaluateCategory.details ?? []),
  ].forEach((detail) => {
    rows.push({
      label: taskTypeLabels[detail.taskType] ?? detail.taskType,
      quota: detail.quota,
      used: detail.used,
      remaining: detail.remaining,
    });
  });
  return rows;
});

function openDetailDialog(row: MemberPageVO) {
  detailDialog.visible = true;
  detailDialog.loading = true;
  detailDialog.userId = row.userId;
  detailDialog.activeTab = "basic";
  resetDetailTabs();
  detailData.value = undefined;
  MemberAPI.getDetail(row.userId)
    .then((data) => {
      detailData.value = data;
    })
    .finally(() => {
      detailDialog.loading = false;
    });
}

function closeDetailDialog() {
  detailDialog.visible = false;
  detailData.value = undefined;
  resetDetailTabs();
}

function resetDetailTabs() {
  Object.assign(growthLogs, {
    loading: false,
    loaded: false,
    list: [],
    total: 0,
    pageNum: 1,
    pageSize: 10,
  });
  Object.assign(consumption, {
    loading: false,
    loaded: false,
    list: [],
    total: 0,
    pageNum: 1,
    pageSize: 10,
  });
  Object.assign(benefitUsage, {
    loading: false,
    loaded: false,
    data: undefined,
  });
  Object.assign(operationLogs, {
    loading: false,
    loaded: false,
    list: [],
    total: 0,
    pageNum: 1,
    pageSize: 10,
  });
}

function loadGrowthLogs() {
  growthLogs.loading = true;
  MemberAPI.getAdminGrowthLogs(detailDialog.userId, {
    pageNum: growthLogs.pageNum,
    pageSize: growthLogs.pageSize,
  })
    .then((data) => {
      growthLogs.list = data.list;
      growthLogs.total = data.total;
      growthLogs.loaded = true;
    })
    .finally(() => {
      growthLogs.loading = false;
    });
}

function loadConsumptionRecords() {
  consumption.loading = true;
  MemberAPI.getConsumptionRecords(detailDialog.userId, {
    pageNum: consumption.pageNum,
    pageSize: consumption.pageSize,
  })
    .then((data) => {
      consumption.list = data.list;
      consumption.total = data.total;
      consumption.loaded = true;
    })
    .finally(() => {
      consumption.loading = false;
    });
}

function loadBenefitUsage() {
  benefitUsage.loading = true;
  MemberAPI.getBenefitUsage(detailDialog.userId)
    .then((data) => {
      benefitUsage.data = data;
      benefitUsage.loaded = true;
    })
    .finally(() => {
      benefitUsage.loading = false;
    });
}

function loadOperationLogs() {
  operationLogs.loading = true;
  MemberAPI.getOperationLogs(detailDialog.userId, {
    pageNum: operationLogs.pageNum,
    pageSize: operationLogs.pageSize,
  })
    .then((data) => {
      operationLogs.list = data.list;
      operationLogs.total = data.total;
      operationLogs.loaded = true;
    })
    .finally(() => {
      operationLogs.loading = false;
    });
}

function handleDetailTabChange(name: string | number) {
  if (name === "growthLogs" && !growthLogs.loaded) {
    loadGrowthLogs();
  } else if (name === "consumption" && !consumption.loaded) {
    loadConsumptionRecords();
  } else if (name === "benefitUsage" && !benefitUsage.loaded) {
    loadBenefitUsage();
  } else if (name === "operationLogs" && !operationLogs.loaded) {
    loadOperationLogs();
  }
}

// ==================== 等级调整 ====================
const levelDialog = reactive({
  visible: false,
  userId: 0,
  username: "",
  currentLevelCode: "level_0" as MemberLevelCode,
  currentLevelName: "",
});

const levelForm = reactive<MemberLevelAdjustForm>({
  levelCode: "level_0",
  expireTime: undefined,
  reason: "",
});

const levelRules = reactive({
  levelCode: [{ required: true, message: "请选择目标等级", trigger: "change" }],
  reason: [
    { required: true, message: "请输入调整原因", trigger: "blur" },
    { min: 2, max: 200, message: "原因长度为2-200字符", trigger: "blur" },
  ],
});

function openLevelDialog(row: MemberPageVO) {
  levelDialog.userId = row.userId;
  levelDialog.username = row.username;
  levelDialog.currentLevelCode = row.levelCode;
  levelDialog.currentLevelName = row.levelName;
  levelForm.levelCode = row.levelCode;
  levelForm.expireTime = undefined;
  levelForm.reason = "";
  levelDialog.visible = true;
}

function resetLevelForm() {
  levelFormRef.value?.resetFields();
  levelFormRef.value?.clearValidate();
}

function submitLevelAdjust() {
  levelFormRef.value.validate((valid: boolean) => {
    if (!valid) return;
    const isDowngrade =
      levelOrderMap[levelDialog.currentLevelCode] >
      levelOrderMap[levelForm.levelCode];
    const targetLabel = levelOptions.find(
      (o) => o.value === levelForm.levelCode
    )?.label;
    const doSubmit = () => {
      loading.value = true;
      MemberAPI.adjustLevel(levelDialog.userId, levelForm)
        .then(() => {
          ElMessage.success("等级调整成功");
          levelDialog.visible = false;
          handleQuery();
        })
        .finally(() => {
          loading.value = false;
        });
    };
    if (isDowngrade) {
      ElMessageBox.confirm(
        `确认将会员「${levelDialog.username}」从 ${levelDialog.currentLevelName} 降级为 ${targetLabel} 吗？`,
        "降级确认",
        {
          confirmButtonText: "确定",
          cancelButtonText: "取消",
          type: "warning",
        }
      )
        .then(() => doSubmit())
        .catch(() => {});
    } else {
      doSubmit();
    }
  });
}

// ==================== 成长值调整 ====================
const growthDialog = reactive({
  visible: false,
  userId: 0,
  username: "",
  currentGrowth: 0,
});

const growthForm = reactive<MemberGrowthAdjustForm>({
  changeValue: 0,
  reason: "",
});

const growthRules = reactive({
  changeValue: [{ required: true, message: "请输入变动值", trigger: "blur" }],
  reason: [{ required: true, message: "请输入调整原因", trigger: "blur" }],
});

const expectedGrowth = computed(() => {
  return growthDialog.currentGrowth + (growthForm.changeValue || 0);
});

function openGrowthDialog(row: MemberPageVO) {
  growthDialog.userId = row.userId;
  growthDialog.username = row.username;
  growthDialog.currentGrowth = row.growthValue;
  growthForm.changeValue = 0;
  growthForm.reason = "";
  growthDialog.visible = true;
}

function resetGrowthForm() {
  growthFormRef.value?.resetFields();
  growthFormRef.value?.clearValidate();
}

function submitGrowthAdjust() {
  growthFormRef.value.validate((valid: boolean) => {
    if (!valid) return;
    loading.value = true;
    MemberAPI.adjustGrowth(growthDialog.userId, growthForm)
      .then(() => {
        ElMessage.success("成长值调整成功");
        growthDialog.visible = false;
        handleQuery();
      })
      .finally(() => {
        loading.value = false;
      });
  });
}

// ==================== 冻结/解冻 ====================
const freezeDialog = reactive({
  visible: false,
  userId: 0,
  username: "",
});

const freezeForm = reactive<MemberStatusForm>({
  status: 0,
  reason: "",
});

const freezeRules = computed(() => {
  if (freezeForm.status === 0) {
    return {
      reason: [{ required: true, message: "请输入冻结原因", trigger: "blur" }],
    };
  }
  return {};
});

function openFreezeDialog(row: MemberPageVO) {
  freezeDialog.userId = row.userId;
  freezeDialog.username = row.username;
  freezeForm.status = row.status === 1 ? 0 : 1;
  freezeForm.reason = "";
  freezeDialog.visible = true;
}

function resetFreezeForm() {
  freezeFormRef.value?.resetFields();
  freezeFormRef.value?.clearValidate();
}

function submitStatusChange() {
  const submit = () => {
    loading.value = true;
    MemberAPI.updateStatus(freezeDialog.userId, {
      status: freezeForm.status,
      reason: freezeForm.status === 0 ? freezeForm.reason : undefined,
    })
      .then(() => {
        ElMessage.success(freezeForm.status === 0 ? "冻结成功" : "解冻成功");
        freezeDialog.visible = false;
        handleQuery();
      })
      .finally(() => {
        loading.value = false;
      });
  };
  if (freezeForm.status === 0) {
    freezeFormRef.value.validate((valid: boolean) => {
      if (!valid) return;
      submit();
    });
  } else {
    submit();
  }
}

// ==================== 权益配置 ====================
const benefitDrawer = reactive({
  visible: false,
  loading: false,
});
const benefitList = ref<BenefitVO[]>([]);

function openBenefitDrawer() {
  benefitDrawer.visible = true;
}

function loadBenefits() {
  benefitDrawer.loading = true;
  MemberAPI.listBenefits()
    .then((data) => {
      benefitList.value = data;
    })
    .finally(() => {
      benefitDrawer.loading = false;
    });
}

function saveBenefit(row: BenefitVO) {
  const payload: BenefitForm = {
    levelName: row.levelName,
    growthMin: row.growthMin,
    growthMax: row.growthMax,
    monthlyDehazeQuota: row.monthlyDehazeQuota,
    monthlyDerainQuota: row.monthlyDerainQuota,
    monthlyDesnowQuota: row.monthlyDesnowQuota,
    monthlyLowlightQuota: row.monthlyLowlightQuota,
    monthlySuperResolutionQuota: row.monthlySuperResolutionQuota,
    monthlyDenoiseQuota: row.monthlyDenoiseQuota,
    monthlyInpaintQuota: row.monthlyInpaintQuota,
    monthlyEvaluateQuota: row.monthlyEvaluateQuota,
    aiCreditsDaily: row.aiCreditsDaily,
    aiCreditsMonthly: row.aiCreditsMonthly,
    multimodalLimit: row.multimodalLimit,
    maxDevices: row.maxDevices,
    vipGiftCredits: row.vipGiftCredits,
    historyRetention: row.historyRetention,
    batchLimit: row.batchLimit,
    priority: row.priority,
    advancedParams: row.advancedParams,
    hdExport: row.hdExport,
    reportExport: row.reportExport,
    batchDownload: row.batchDownload,
    sort: row.sort,
    status: row.status,
  };
  MemberAPI.updateBenefit(row.levelCode, payload).then(() => {
    ElMessage.success(`「${row.levelName}」权益配置已保存`);
  });
}

onMounted(() => {
  handleQuery();
});
</script>

<style lang="scss" scoped>
.text-secondary {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
