package com.pei.dehaze.service.impl;

import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.config.WebSocketMessageRelay;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.model.query.TaskQuery;
import com.pei.dehaze.model.vo.TaskVO;
import com.pei.dehaze.service.TaskService;
import com.pei.dehaze.service.TaskExecutor;
import com.pei.dehaze.security.util.SecurityUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * 任务服务实现
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class TaskServiceImpl extends ServiceImpl<SysTaskMapper, SysTask> implements TaskService {

    private final TaskExecutor taskExecutor;

    private final RedisTemplate<String, Object> redisTemplate;

    private final WebSocketMessageRelay wsMessageRelay;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public TaskVO createTask(ExportTaskCreateForm form, String idempotencyKey) {
        Long currentUserId = SecurityUtils.getUserId();
        if (currentUserId == null) {
            throw new BusinessException("用户未登录");
        }

        // 幂等去重：相同 idempotencyKey 直接返回已有任务
        if (StrUtil.isNotBlank(idempotencyKey)) {
            SysTask existingTask = this.getOne(new LambdaQueryWrapper<SysTask>()
                    .eq(SysTask::getIdempotencyKey, idempotencyKey));
            if (existingTask != null) {
                log.info("幂等键命中，返回已有任务: taskId={}, idempotencyKey={}",
                        existingTask.getTaskId(), idempotencyKey);
                return convertToTaskVO(existingTask);
            }
        }

        String taskId = IdUtil.simpleUUID();

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setTaskType(form.getType());
        sysTask.setStatus(TaskConstants.STATUS_PENDING);
        sysTask.setProgress(0);
        sysTask.setTotalFiles(0);
        sysTask.setProcessedFiles(0);
        sysTask.setParams(JSONUtil.toJsonStr(form));
        sysTask.setCreateBy(currentUserId);
        sysTask.setStartedAt(null);
        sysTask.setCompletedAt(null);
        sysTask.setExpiresAt(LocalDateTime.now().plusSeconds(TaskConstants.TASK_EXPIRE_SECONDS));
        sysTask.setIdempotencyKey(idempotencyKey);
        sysTask.setRetryCount(0);

        this.save(sysTask);

        // 缓存幂等键映射
        if (StrUtil.isNotBlank(idempotencyKey)) {
            String idempotencyRedisKey = TaskConstants.IDEMPOTENCY_KEY_PREFIX + idempotencyKey;
            redisTemplate.opsForValue().set(idempotencyRedisKey, taskId,
                    TaskConstants.IDEMPOTENCY_KEY_EXPIRE_SECONDS, TimeUnit.SECONDS);
        }

        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

        TaskVO taskVO = convertToTaskVO(sysTask);

        // 事务提交后再发布任务，避免异步线程读不到未提交的数据
        Long taskIdForAsync = sysTask.getId();
        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    taskExecutor.publishExportTask(taskIdForAsync);
                }
            });
        } else {
            // 无活跃事务同步（如单元测试），直接发布
            taskExecutor.publishExportTask(taskIdForAsync);
        }

        log.info("创建任务成功: taskId={}, type={}, userId={}", taskId, form.getType(), currentUserId);

        return taskVO;
    }

    @Override
    public TaskVO getTaskStatus(String taskId) {
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("任务ID不能为空");
        }

        SysTask sysTask = getTaskEntity(taskId);
        if (sysTask == null) {
            return null;
        }

        log.info("查询任务状态: taskId={}, status={}", taskId, sysTask.getStatus());

        return convertToTaskVO(sysTask);
    }

    @Override
    public String getDownloadUrl(String taskId) {
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("任务ID不能为空");
        }

        SysTask sysTask = getTaskEntity(taskId);
        if (sysTask == null) {
            throw new BusinessException("任务不存在: " + taskId);
        }

        if (sysTask.getStatus() == null || sysTask.getStatus() != TaskConstants.STATUS_COMPLETED) {
            throw new BusinessException("任务未完成，无法下载: status=" + sysTask.getStatus());
        }

        if (sysTask.getExpiresAt() != null && sysTask.getExpiresAt().isBefore(LocalDateTime.now())) {
            throw new BusinessException("任务已过期，无法下载: expiresAt=" + sysTask.getExpiresAt());
        }

        if (StrUtil.isBlank(sysTask.getResult())) {
            throw new BusinessException("任务结果为空: " + taskId);
        }

        String downloadUrl = sysTask.getResult();
        log.info("生成下载链接: taskId={}, url={}", taskId, downloadUrl);

        return downloadUrl;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void cancelTask(String taskId) {
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("任务ID不能为空");
        }

        SysTask sysTask = getTaskEntity(taskId);
        if (sysTask == null) {
            throw new BusinessException("任务不存在: " + taskId);
        }

        if (sysTask.getStatus() != null
                && (sysTask.getStatus() == TaskConstants.STATUS_COMPLETED
                || sysTask.getStatus() == TaskConstants.STATUS_FAILED)) {
            throw new BusinessException("任务已完成或失败，无法取消: status=" + sysTask.getStatus());
        }

        if (sysTask.getStatus() != null && sysTask.getStatus() == TaskConstants.STATUS_CANCELLED) {
            throw new BusinessException("任务已取消: " + taskId);
        }

        sysTask.setStatus(TaskConstants.STATUS_CANCELLED);
        sysTask.setCompletedAt(LocalDateTime.now());

        this.updateById(sysTask);

        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

        String cancelKey = TaskConstants.TASK_CANCEL_PREFIX + taskId;
        redisTemplate.opsForValue().set(cancelKey, true, TaskConstants.CANCEL_FLAG_EXPIRE_SECONDS, TimeUnit.SECONDS);

        // WebSocket 推送取消通知（通过 Redis Pub/Sub 跨实例投递，对齐 Python 消息格式）
        try {
            Map<String, Object> message = new HashMap<>();
            message.put("type", "task_status");
            message.put("task_id", sysTask.getTaskId());
            message.put("status", TaskConstants.STATUS_CANCELLED);
            message.put("progress", sysTask.getProgress());
            message.put("result", null);
            message.put("error_message", null);
            message.put("timestamp", LocalDateTime.now().toString());
            wsMessageRelay.publishToUser(sysTask.getCreateBy(), WebSocketMessageRelay.DEST_TASK, message);
        } catch (Exception e) {
            log.warn("WebSocket 推送取消通知失败（不影响任务执行）: taskId={}", taskId, e);
        }

        log.info("取消任务成功: taskId={}", taskId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public TaskVO retryTask(String taskId) {
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("任务ID不能为空");
        }

        Long currentUserId = SecurityUtils.getUserId();
        if (currentUserId == null) {
            throw new BusinessException("用户未登录");
        }

        SysTask sysTask = getTaskEntity(taskId);
        if (sysTask == null) {
            throw new BusinessException("任务不存在: " + taskId);
        }

        if (!sysTask.getCreateBy().equals(currentUserId)) {
            throw new BusinessException("无权操作他人任务");
        }

        if (sysTask.getStatus() == null || sysTask.getStatus() != TaskConstants.STATUS_FAILED) {
            throw new BusinessException("仅失败任务可重试");
        }

        // 解析原始参数
        ExportTaskCreateForm form = JSONUtil.toBean(sysTask.getParams(), ExportTaskCreateForm.class);

        // 重置任务状态，递增重试次数
        sysTask.setStatus(TaskConstants.STATUS_PENDING);
        sysTask.setProgress(0);
        sysTask.setProcessedFiles(0);
        sysTask.setErrorMessage(null);
        sysTask.setStartedAt(null);
        sysTask.setCompletedAt(null);
        sysTask.setRetryCount(sysTask.getRetryCount() + 1);
        sysTask.setWorkerId(null);
        sysTask.setExpiresAt(LocalDateTime.now().plusSeconds(TaskConstants.TASK_EXPIRE_SECONDS));
        this.updateById(sysTask);

        // 更新缓存
        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

        // 事务提交后再重新发布任务，避免异步线程读不到未提交的数据
        Long taskIdForAsync = sysTask.getId();
        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    taskExecutor.publishExportTask(taskIdForAsync);
                }
            });
        } else {
            // 无活跃事务同步（如单元测试），直接发布
            taskExecutor.publishExportTask(taskIdForAsync);
        }

        log.info("重试任务: taskId={}, userId={}, retryCount={}", taskId, currentUserId, sysTask.getRetryCount());
        return convertToTaskVO(sysTask);
    }

    @Override
    public PageResult<TaskVO> listMyTasks(int pageNum, int pageSize) {
        return listMyTasks(buildQuery(pageNum, pageSize, null, null, null));
    }

    @Override
    public PageResult<TaskVO> listMyTasks(TaskQuery query) {
        Long currentUserId = SecurityUtils.getUserId();
        if (currentUserId == null) {
            throw new BusinessException("用户未登录");
        }

        LambdaQueryWrapper<SysTask> wrapper = new LambdaQueryWrapper<SysTask>()
                .eq(SysTask::getCreateBy, currentUserId);

        if (StrUtil.isNotBlank(query.getTaskType())) {
            List<String> types = Arrays.stream(query.getTaskType().split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toList());
            if (types.size() == 1) {
                wrapper.eq(SysTask::getTaskType, types.get(0));
            } else if (!types.isEmpty()) {
                wrapper.in(SysTask::getTaskType, types);
            }
        }

        if (query.getStatus() != null) {
            wrapper.eq(SysTask::getStatus, query.getStatus());
        }

        if (StrUtil.isNotBlank(query.getTaskCategory())) {
            List<String> typesOfCategory = resolveTypesByCategory(query.getTaskCategory());
            if (typesOfCategory != null) {
                if (typesOfCategory.isEmpty()) {
                    wrapper.eq(SysTask::getTaskType, "__none__");
                } else {
                    wrapper.in(SysTask::getTaskType, typesOfCategory);
                }
            }
        }

        wrapper.orderByDesc(SysTask::getCreateTime);

        Page<SysTask> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysTask> taskPage = this.page(page, wrapper);
        IPage<TaskVO> voPage = taskPage.convert(this::convertToTaskVO);

        log.info("查询用户任务列表: userId={}, total={}", currentUserId, taskPage.getTotal());

        return PageResult.success(voPage);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void updateTaskCompleted(String taskId, String result, LocalDateTime expiresAt) {
        SysTask sysTask = getTaskEntity(taskId);
        if (sysTask == null) {
            throw new BusinessException("任务不存在: " + taskId);
        }
        sysTask.setStatus(TaskConstants.STATUS_COMPLETED);
        sysTask.setProgress(100);
        sysTask.setResult(result);
        sysTask.setCompletedAt(LocalDateTime.now());
        sysTask.setExpiresAt(expiresAt);
        this.updateById(sysTask);

        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

        pushTaskStatusMessage(sysTask, TaskConstants.STATUS_COMPLETED, result, null);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void updateTaskFailed(String taskId, String errorMessage) {
        SysTask sysTask = getTaskEntity(taskId);
        if (sysTask == null) {
            throw new BusinessException("任务不存在: " + taskId);
        }
        sysTask.setStatus(TaskConstants.STATUS_FAILED);
        sysTask.setErrorMessage(errorMessage);
        sysTask.setCompletedAt(LocalDateTime.now());
        this.updateById(sysTask);

        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

        pushTaskStatusMessage(sysTask, TaskConstants.STATUS_FAILED, null, errorMessage);
    }

    private TaskQuery buildQuery(int pageNum, int pageSize, String taskType, Integer status, String taskCategory) {
        TaskQuery query = new TaskQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setTaskType(taskType);
        query.setStatus(status);
        query.setTaskCategory(taskCategory);
        return query;
    }

    /**
     * 根据任务类别反查任务类型列表（用于按类别筛选）
     * <p>由于 sys_task 表无 task_category 字段，这里通过类型后缀推断。
     */
    private List<String> resolveTypesByCategory(String category) {
        if (TaskConstants.CATEGORY_IMPORT.equalsIgnoreCase(category)) {
            return List.of(
                    TaskConstants.TYPE_USER_IMPORT, TaskConstants.TYPE_ROLE_IMPORT,
                    TaskConstants.TYPE_DEPT_IMPORT, TaskConstants.TYPE_MENU_IMPORT,
                    TaskConstants.TYPE_DICT_IMPORT, TaskConstants.TYPE_ALGORITHM_IMPORT
            );
        }
        if (TaskConstants.CATEGORY_EXPORT.equalsIgnoreCase(category)) {
            return List.of(
                    TaskConstants.TYPE_DATASET_EXPORT,
                    TaskConstants.TYPE_USER_EXPORT, TaskConstants.TYPE_ROLE_EXPORT,
                    TaskConstants.TYPE_DEPT_EXPORT, TaskConstants.TYPE_MENU_EXPORT,
                    TaskConstants.TYPE_DICT_EXPORT, TaskConstants.TYPE_ALGORITHM_EXPORT
            );
        }
        return null;
    }

    private void pushTaskStatusMessage(SysTask task, Integer status, String result, String errorMessage) {
        try {
            Map<String, Object> message = new HashMap<>();
            message.put("type", "task_status");
            message.put("task_id", task.getTaskId());
            message.put("status", status);
            message.put("progress", task.getProgress());
            message.put("result", result);
            message.put("error_message", errorMessage);
            message.put("timestamp", LocalDateTime.now().toString());
            wsMessageRelay.publishToUser(task.getCreateBy(), WebSocketMessageRelay.DEST_TASK, message);
        } catch (Exception e) {
            log.warn("WebSocket 推送失败（不影响任务执行）: {}", e.getMessage());
        }
    }

    private SysTask getTaskEntity(String taskId) {
        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        SysTask cachedTask = (SysTask) redisTemplate.opsForValue().get(cacheKey);

        if (cachedTask != null) {
            return cachedTask;
        }

        return this.getOne(new LambdaQueryWrapper<SysTask>()
                .eq(SysTask::getTaskId, taskId));
    }

    private TaskVO convertToTaskVO(SysTask sysTask) {
        TaskVO taskVO = new TaskVO();
        taskVO.setTaskId(sysTask.getTaskId());
        taskVO.setTaskType(sysTask.getTaskType());
        taskVO.setTaskCategory(TaskConstants.getCategoryByType(sysTask.getTaskType()));
        taskVO.setStatus(sysTask.getStatus());
        taskVO.setProgress(sysTask.getProgress());
        taskVO.setTotalFiles(sysTask.getTotalFiles());
        taskVO.setProcessedFiles(sysTask.getProcessedFiles());

        if (sysTask.getStatus() != null
                && sysTask.getStatus() == TaskConstants.STATUS_COMPLETED
                && StrUtil.isNotBlank(sysTask.getResult())) {
            taskVO.setDownloadUrl(sysTask.getResult());
        }

        taskVO.setExpiresAt(sysTask.getExpiresAt());
        taskVO.setCreatedAt(sysTask.getCreateTime());
        taskVO.setStartedAt(sysTask.getStartedAt());
        taskVO.setCompletedAt(sysTask.getCompletedAt());

        if (StrUtil.isNotBlank(sysTask.getErrorMessage())) {
            taskVO.setError(sysTask.getErrorMessage());
        }

        taskVO.setIdempotencyKey(sysTask.getIdempotencyKey());
        taskVO.setRetryCount(sysTask.getRetryCount());
        taskVO.setWorkerId(sysTask.getWorkerId());

        return taskVO;
    }
}
