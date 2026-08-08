package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.ClientLogBatchForm;
import com.pei.dehaze.plugin.ratelimit.annotation.RateLimit;
import com.pei.dehaze.service.ClientLogService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 前端日志接收接口。
 * <p>
 * 接收前端 SDK 批量上报的日志，复用各端现有限流（@RateLimit，IP 维度，60 秒 1000 次），
 * 落盘 logs/{yyyy-MM-dd}/client.log 供 filebeat 采集。匿名也允许上报（ERROR 级别），
 * 已登录用户从会话解析 user_id 注入。
 */
@Tag(name = "99.前端日志")
@RestController
@RequestMapping("/api/v1/logs/client")
@RequiredArgsConstructor
public class ClientLogController {

    private final ClientLogService clientLogService;

    @Operation(summary = "前端日志批量上报")
    @RateLimit(
        key = "rate:limit:client-log:", 
        timeWindow = 60, maxRequests = 1000,
        type = RateLimit.LimitType.IP, 
        limiter = RateLimit.LimiterType.FIXED_WINDOW,
        message = "日志上报过于频繁，请稍后再试"
    )
    @PostMapping
    public Result<Void> collect(@RequestBody @Valid ClientLogBatchForm form) {
        clientLogService.collect(form);
        return Result.success();
    }
}
