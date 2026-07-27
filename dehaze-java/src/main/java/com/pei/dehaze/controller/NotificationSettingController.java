package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.NotificationSettingForm;
import com.pei.dehaze.model.vo.NotificationSettingsVO;
import com.pei.dehaze.service.NotificationSettingService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Tag(name = "10.消息通知-通知设置")
@RestController
@RequestMapping("/api/v1/notification-settings")
@RequiredArgsConstructor
public class NotificationSettingController {

    private final NotificationSettingService notificationSettingService;

    @Operation(summary = "获取通知偏好设置")
    @GetMapping
    public Result<NotificationSettingsVO> get() {
        return Result.success(notificationSettingService.get());
    }

    @Operation(summary = "更新通知偏好设置")
    @PutMapping
    public Result<Void> update(@RequestBody NotificationSettingForm form) {
        notificationSettingService.update(form);
        return Result.success();
    }
}
