package com.pei.dehaze.controller;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.form.CompareReportForm;
import com.pei.dehaze.model.vo.CompareReportResultVO;
import com.pei.dehaze.service.CompareService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@Tag(name = "19.效果对比")
@RestController
@RequestMapping("/api/v1/compare")
@RequiredArgsConstructor
public class CompareController {

    private final CompareService compareService;
    private final SysEvalLogMapper evalLogMapper;

    @Operation(summary = "生成对比报告（异步任务）")
    @PostMapping("/report")
    public Result<CompareReportResultVO> generateReport(@Valid @RequestBody CompareReportForm form) {
        CompareReportResultVO result = compareService.generateReport(form);
        return Result.success(result);
    }

    @Operation(summary = "查询对比报告状态/下载对比报告")
    @GetMapping("/report/{taskId}")
    public ResponseEntity<?> getOrDownloadReport(
            @Parameter(description = "报告任务ID") @PathVariable Long taskId,
            @Parameter(description = "下载标识（true时返回HTML文件流，否则返回JSON状态）")
            @RequestParam(required = false, defaultValue = "false") boolean download) {
        SysEvalLog reportTask = evalLogMapper.selectById(taskId);
        if (reportTask == null) {
            return ResponseEntity.ok(Result.failed(ResultCode.RESOURCE_NOT_FOUND, "报告不存在"));
        }

        if (!download) {
            CompareReportResultVO vo = new CompareReportResultVO();
            vo.setTaskId(reportTask.getId());
            vo.setStatus(reportTask.getStatus());
            if (reportTask.getStatus() == LogStatusEnum.FAILED) {
                vo.setErrorMessage(reportTask.getErrorMessage());
            } else if (reportTask.getStatus() == LogStatusEnum.COMPLETED) {
                vo.setDownloadUrl("/api/v1/compare/report/" + reportTask.getId() + "?download=true");
            }
            return ResponseEntity.ok(Result.success(vo));
        }

        if (reportTask.getStatus() == LogStatusEnum.PROCESSING) {
            CompareReportResultVO vo = new CompareReportResultVO();
            vo.setTaskId(reportTask.getId());
            vo.setStatus(LogStatusEnum.PROCESSING);
            return ResponseEntity.ok(Result.success(vo));
        }
        if (reportTask.getStatus() == LogStatusEnum.FAILED) {
            return ResponseEntity.ok(Result.failed(ResultCode.SYSTEM_EXECUTION_ERROR,
                    "报告生成失败：" + (reportTask.getErrorMessage() != null ? reportTask.getErrorMessage() : "未知错误")));
        }

        String resultJson = reportTask.getResult();
        if (resultJson == null || resultJson.isBlank()) {
            return ResponseEntity.ok(Result.failed(ResultCode.RESOURCE_NOT_FOUND, "报告内容为空"));
        }

        JSONObject result = JSONUtil.parseObj(resultJson);
        String reportHtml = result.getStr("reportHtml");
        if (reportHtml == null) {
            return ResponseEntity.ok(Result.failed(ResultCode.RESOURCE_NOT_FOUND, "报告内容为空"));
        }

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=compare-report.html")
                .contentType(MediaType.TEXT_HTML)
                .body(reportHtml);
    }
}
