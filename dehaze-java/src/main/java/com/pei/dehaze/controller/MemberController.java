package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.BenefitForm;
import com.pei.dehaze.model.form.MemberGrowthAdjustForm;
import com.pei.dehaze.model.form.MemberLevelAdjustForm;
import com.pei.dehaze.model.form.MemberStatusForm;
import com.pei.dehaze.model.query.GrowthLogQuery;
import com.pei.dehaze.model.query.MemberPageQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.service.MemberBenefitService;
import com.pei.dehaze.service.MemberService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "11.会员管理")
@RestController
@RequestMapping("/api/v1/members")
@RequiredArgsConstructor
public class MemberController {

    private final MemberService memberService;
    private final MemberBenefitService memberBenefitService;

    @Operation(summary = "当前用户会员信息")
    @GetMapping("/profile")
    public Result<MemberProfileVO> getProfile() {
        return Result.success(memberService.getProfile());
    }

    @Operation(summary = "成长值变动明细")
    @GetMapping("/growth-logs")
    public PageResult<GrowthLogVO> getGrowthLogs(@ParameterObject GrowthLogQuery query) {
        Page<GrowthLogVO> page = memberService.getGrowthLogs(query);
        return PageResult.success(page);
    }

    @Operation(summary = "每日签到")
    @PostMapping("/sign-in")
    public Result<SignInResultVO> signIn() {
        return Result.success(memberService.signIn());
    }

    @Operation(summary = "签到日历")
    @GetMapping("/sign-in/calendar")
    public Result<SignInCalendarVO> getSignInCalendar(
            @Parameter(description = "年份", required = true) @RequestParam Integer year,
            @Parameter(description = "月份(1-12)", required = true) @RequestParam Integer month) {
        return Result.success(memberService.getSignInCalendar(year, month));
    }

    @Operation(summary = "会员分页列表")
    @GetMapping("/page")
    public PageResult<MemberPageVO> getPage(@ParameterObject MemberPageQuery query) {
        Page<MemberPageVO> page = memberService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "会员详情")
    @GetMapping("/{userId}")
    public Result<MemberDetailVO> getDetail(@Parameter(description = "用户ID") @PathVariable Long userId) {
        return Result.success(memberService.getDetail(userId));
    }

    @Operation(summary = "等级调整")
    @PutMapping("/{userId}/level")
    @PreAuthorize("@ss.hasPerm('member:level:edit')")
    public Result<Void> adjustLevel(@Parameter(description = "用户ID") @PathVariable Long userId,
                                    @Valid @RequestBody MemberLevelAdjustForm form) {
        memberService.adjustLevel(userId, form);
        return Result.success();
    }

    @Operation(summary = "成长值调整")
    @PutMapping("/{userId}/growth")
    @PreAuthorize("@ss.hasPerm('member:growth:edit')")
    public Result<Void> adjustGrowth(@Parameter(description = "用户ID") @PathVariable Long userId,
                                     @Valid @RequestBody MemberGrowthAdjustForm form) {
        memberService.adjustGrowth(userId, form);
        return Result.success();
    }

    @Operation(summary = "冻结/解冻")
    @PutMapping("/{userId}/status")
    @PreAuthorize("@ss.hasPerm('member:status:edit')")
    public Result<Void> updateStatus(@Parameter(description = "用户ID") @PathVariable Long userId,
                                     @Valid @RequestBody MemberStatusForm form) {
        memberService.updateStatus(userId, form);
        return Result.success();
    }

    @Operation(summary = "权益配置列表")
    @GetMapping("/benefits")
    public Result<List<BenefitVO>> listBenefits() {
        return Result.success(memberBenefitService.listVOs());
    }

    @Operation(summary = "修改权益配置")
    @PutMapping("/benefits/{level}")
    @PreAuthorize("@ss.hasPerm('member:benefit:edit')")
    public Result<Void> updateBenefit(@Parameter(description = "等级标识") @PathVariable String level,
                                      @Valid @RequestBody BenefitForm form) {
        memberBenefitService.updateByLevelCode(level, form);
        return Result.success();
    }
}
