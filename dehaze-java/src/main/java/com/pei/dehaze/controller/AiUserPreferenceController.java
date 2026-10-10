package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiUserPreferenceForm;
import com.pei.dehaze.model.vo.AiUserPreferenceVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiUserPreferenceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 用户偏好默认对话模型（登录即可，无权限码）。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话-用户偏好")
@RestController
@RequestMapping("/api/v1/ai/user-preference")
@RequiredArgsConstructor
public class AiUserPreferenceController {

    private final AiUserPreferenceService userPreferenceService;

    @Operation(summary = "获取用户偏好默认模型")
    @GetMapping
    public Result<AiUserPreferenceVO> get() {
        return Result.success(userPreferenceService.get(SecurityUtils.getUserId()));
    }

    @Operation(summary = "设置用户偏好默认模型（null 即清除）")
    @PutMapping
    public Result<Void> update(@RequestBody AiUserPreferenceForm form) {
        userPreferenceService.update(SecurityUtils.getUserId(), form);
        return Result.success();
    }
}
