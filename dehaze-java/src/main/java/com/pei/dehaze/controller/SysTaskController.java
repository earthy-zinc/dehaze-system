package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.model.query.TaskQuery;
import com.pei.dehaze.model.vo.TaskVO;
import com.pei.dehaze.service.TaskService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * 统一任务控制器
 */
@Tag(name = "10.任务接口")
@RestController
@RequestMapping("/api/v1/tasks")
@RequiredArgsConstructor
public class SysTaskController {

    private final TaskService taskService;

    @PostMapping
    @Operation(
        summary = "创建任务",
        description = "创建异步任务，支持多种任务类型：dataset_export（数据集导出）、item_download（数据项下载）、batch_download（批量下载）"
    )
    public Result<TaskVO> createTask(
            @Valid @RequestBody ExportTaskCreateForm form,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey
    ) {
        TaskVO task = taskService.createTask(form, idempotencyKey);
        return Result.success(task);
    }

    @GetMapping("/{taskId}")
    @Operation(
        summary = "查询任务状态",
        description = "根据任务ID查询任务的当前状态和进度信息"
    )
    public Result<TaskVO> getTaskStatus(
        @Parameter(description = "任务ID", required = true)
        @PathVariable String taskId
    ) {
        TaskVO task = taskService.getTaskStatus(taskId);
        return Result.success(task);
    }

    @GetMapping("/{taskId}/download")
    @Operation(
        summary = "下载任务结果文件",
        description = "下载已完成任务的结果文件（ZIP格式）"
    )
    public ResponseEntity<Resource> downloadFile(
        @Parameter(description = "任务ID", required = true)
        @PathVariable String taskId
    ) {
        String downloadUrl = taskService.getDownloadUrl(taskId);
        if (downloadUrl == null) {
            return ResponseEntity.notFound().build();
        }

        // 重定向到文件下载URL
        return ResponseEntity.status(HttpStatus.FOUND)
            .header(HttpHeaders.LOCATION, downloadUrl)
            .build();
    }

    @DeleteMapping("/{taskId}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    @Operation(
        summary = "取消任务",
        description = "取消正在执行的任务，只能取消自己创建的任务"
    )
    public void cancelTask(
        @Parameter(description = "任务ID", required = true)
        @PathVariable String taskId
    ) {
        taskService.cancelTask(taskId);
    }

    @PostMapping("/{taskId}/retry")
    @Operation(
        summary = "重试失败的任务",
        description = "仅允许 FAILED 状态的任务重试，重置重试次数后重新提交执行"
    )
    public Result<TaskVO> retryTask(
        @Parameter(description = "任务ID", required = true)
        @PathVariable String taskId
    ) {
        TaskVO task = taskService.retryTask(taskId);
        return Result.success(task);
    }

    @GetMapping
    @Operation(
        summary = "分页查询任务列表",
        description = "查询当前用户的任务列表，支持分页"
    )
    public PageResult<TaskVO> listMyTasks(@ParameterObject TaskQuery query) {
        return taskService.listMyTasks(query.getPageNum(), query.getPageSize());
    }
}
