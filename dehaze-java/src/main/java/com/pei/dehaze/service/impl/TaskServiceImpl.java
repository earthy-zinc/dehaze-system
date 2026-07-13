package com.pei.dehaze.service.impl;

import cn.hutool.core.lang.Assert;
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
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.model.vo.TaskVO;
import com.pei.dehaze.service.TaskService;
import com.pei.dehaze.service.TaskExecutor;
import com.pei.dehaze.security.util.SecurityUtils;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.concurrent.TimeUnit;

/**
 * 任务服务实现
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Slf4j
@Service
public class TaskServiceImpl extends ServiceImpl<SysTaskMapper, SysTask> implements TaskService {

    @Resource
    private TaskExecutor taskExecutor;

    @Resource
    private RedisTemplate<String, Object> redisTemplate;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public TaskVO createTask(ExportTaskCreateForm form) {
        Long currentUserId = SecurityUtils.getUserId();
        if (currentUserId == null) {
            throw new BusinessException("用户未登录");
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
        sysTask.setCreatedBy(currentUserId);
        sysTask.setStartedAt(null);
        sysTask.setCompletedAt(null);
        sysTask.setExpiresAt(LocalDateTime.now().plusSeconds(TaskConstants.TASK_EXPIRE_SECONDS));

        this.save(sysTask);

        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

        TaskVO taskVO = convertToTaskVO(sysTask);

        taskExecutor.submitExportTask(sysTask.getId(), form);

        log.info("创建任务成功: taskId={}, type={}, userId={}", taskId, form.getType(), currentUserId);

        return taskVO;
    }

    @Override
    public TaskVO getTaskStatus(String taskId) {
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("任务ID不能为空");
        }

        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        SysTask cachedTask = (SysTask) redisTemplate.opsForValue().get(cacheKey);

        if (cachedTask != null) {
            log.debug("从Redis缓存查询任务状态: taskId={}", taskId);
            return convertToTaskVO(cachedTask);
        }

        SysTask sysTask = this.getOne(new LambdaQueryWrapper<SysTask>()
                .eq(SysTask::getTaskId, taskId));

        if (sysTask == null) {
            log.warn("任务不存在: taskId={}", taskId);
            return null;
        }

        redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

        log.info("查询任务状态: taskId={}, status={}", taskId, sysTask.getStatus());

        return convertToTaskVO(sysTask);
    }

    @Override
    public String getDownloadUrl(String taskId) {
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("任务ID不能为空");
        }

        SysTask sysTask = getTaskEntity(taskId);
        Assert.notNull(sysTask, "任务不存在");

        if (!TaskConstants.STATUS_COMPLETED.equals(sysTask.getStatus())) {
            log.warn("任务未完成，无法下载: taskId={}, status={}", taskId, sysTask.getStatus());
            return null;
        }

        if (sysTask.getExpiresAt() != null && sysTask.getExpiresAt().isBefore(LocalDateTime.now())) {
            log.warn("任务已过期，无法下载: taskId={}, expiresAt={}", taskId, sysTask.getExpiresAt());
            return null;
        }

        if (StrUtil.isBlank(sysTask.getResult())) {
            log.warn("任务结果为空: taskId={}", taskId);
            return null;
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
        Assert.notNull(sysTask, "任务不存在");

        if (TaskConstants.STATUS_COMPLETED.equals(sysTask.getStatus()) ||
            TaskConstants.STATUS_FAILED.equals(sysTask.getStatus())) {
            log.warn("任务已完成或失败，无法取消: taskId={}, status={}", taskId, sysTask.getStatus());
            return;
        }

        if (TaskConstants.STATUS_CANCELLED.equals(sysTask.getStatus())) {
            log.warn("任务已取消: taskId={}", taskId);
            return;
        }

        sysTask.setStatus(TaskConstants.STATUS_CANCELLED);
        sysTask.setCompletedAt(LocalDateTime.now());

        this.updateById(sysTask);

        String cacheKey = TaskConstants.TASK_CACHE_PREFIX + taskId;
        redisTemplate.opsForValue().set(cacheKey, sysTask, TaskConstants.TASK_EXPIRE_SECONDS, TimeUnit.SECONDS);

        String cancelKey = TaskConstants.TASK_CANCEL_PREFIX + taskId;
        redisTemplate.opsForValue().set(cancelKey, true, TaskConstants.CANCEL_FLAG_EXPIRE_SECONDS, TimeUnit.SECONDS);

        log.info("取消任务成功: taskId={}", taskId);
    }

    @Override
    public PageResult<TaskVO> listMyTasks(Integer pageNum, Integer pageSize) {
        Long currentUserId = SecurityUtils.getUserId();
        if (currentUserId == null) {
            throw new BusinessException("用户未登录");
        }

        if (pageNum == null || pageNum < 1) {
            pageNum = 1;
        }
        if (pageSize == null || pageSize < 1 || pageSize > 100) {
            pageSize = 10;
        }

        Page<SysTask> page = new Page<>(pageNum, pageSize);
        IPage<SysTask> taskPage = this.page(page, new LambdaQueryWrapper<SysTask>()
                .eq(SysTask::getCreatedBy, currentUserId)
                .orderByDesc(SysTask::getCreateTime));

        IPage<TaskVO> voPage = taskPage.convert(this::convertToTaskVO);

        log.info("查询用户任务列表: userId={}, total={}", currentUserId, taskPage.getTotal());

        return PageResult.success(voPage);
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
        taskVO.setStatus(sysTask.getStatus());
        taskVO.setProgress(sysTask.getProgress());
        taskVO.setTotalFiles(sysTask.getTotalFiles());
        taskVO.setProcessedFiles(sysTask.getProcessedFiles());

        if (TaskConstants.STATUS_COMPLETED.equals(sysTask.getStatus()) &&
            StrUtil.isNotBlank(sysTask.getResult())) {
            taskVO.setDownloadUrl(sysTask.getResult());
        }

        taskVO.setExpiresAt(sysTask.getExpiresAt());
        taskVO.setCreatedAt(sysTask.getCreateTime());
        taskVO.setStartedAt(sysTask.getStartedAt());
        taskVO.setCompletedAt(sysTask.getCompletedAt());

        if (StrUtil.isNotBlank(sysTask.getErrorMessage())) {
            taskVO.setError(sysTask.getErrorMessage());
        }

        return taskVO;
    }
}
