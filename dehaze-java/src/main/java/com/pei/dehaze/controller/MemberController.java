package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.form.BenefitForm;
import com.pei.dehaze.model.form.MemberGrowthAdjustForm;
import com.pei.dehaze.model.form.MemberLevelAdjustForm;
import com.pei.dehaze.model.form.MemberStatusForm;
import com.pei.dehaze.model.query.GrowthLogQuery;
import com.pei.dehaze.model.query.MemberPageQuery;
import com.pei.dehaze.model.query.MyOrderQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.security.service.PermissionService;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.MemberBenefitService;
import com.pei.dehaze.service.MemberService;
import com.pei.dehaze.service.OrderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@Tag(name = "11.会员管理")
@RestController
@RequestMapping("/api/v1/members")
@RequiredArgsConstructor
public class MemberController {

    private final MemberService memberService;
    private final MemberBenefitService memberBenefitService;
    private final OrderService orderService;
    private final PermissionService permissionService;

    @Operation(summary = "当前用户会员信息")
    @GetMapping("/profile")
    public Result<MemberProfileVO> getProfile() {
        return Result.success(memberService.getProfile());
    }

    @Operation(summary = "成长值变动明细")
    @GetMapping("/growth-logs")
    public PageResult<GrowthLogVO> getGrowthLogs(@Valid @ParameterObject GrowthLogQuery query) {
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

    @Operation(summary = "当前用户权益概览")
    @GetMapping("/benefit-summary")
    public Result<Map<String, Object>> getBenefitSummary() {
        return Result.success(memberService.getBenefitSummary(SecurityUtils.getUserId()));
    }

    @Operation(summary = "当前用户试用引导状态")
    @GetMapping("/trial-status")
    public Result<MemberTrialStatusVO> getTrialStatus() {
        return Result.success(memberService.getTrialStatus(SecurityUtils.getUserId()));
    }

    @Operation(summary = "会员分页列表")
    @GetMapping("/page")
    @PreAuthorize("@ss.hasPerm('member:list')")
    public PageResult<MemberPageVO> getPage(@Valid @ParameterObject MemberPageQuery query) {
        Page<MemberPageVO> page = memberService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "会员详情")
    @GetMapping("/{userId}")
    public Result<MemberDetailVO> getDetail(@Parameter(description = "用户ID") @PathVariable Long userId) {
        // 默认仅本人可见；持 member:list 权限可查任意会员（管理端详情弹窗入口）
        Long currentUserId = SecurityUtils.getUserId();
        if (!currentUserId.equals(userId) && !SecurityUtils.isRoot() && !permissionService.hasPerm("member:list")) {
            throw new BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权查看他人会员详情");
        }
        return Result.success(memberService.getDetail(userId));
    }

    @Operation(summary = "会员成长值流水（管理端）")
    @GetMapping("/{userId}/growth-logs")
    @PreAuthorize("@ss.hasPerm('member:list')")
    public PageResult<GrowthLogVO> getMemberGrowthLogs(@Parameter(description = "用户ID") @PathVariable Long userId,
                                                       @Valid @ParameterObject GrowthLogQuery query) {
        return PageResult.success(memberService.getGrowthLogsByUser(userId, query));
    }

    @Operation(summary = "会员消费记录（管理端）")
    @GetMapping("/{userId}/consumption-records")
    @PreAuthorize("@ss.hasPerm('member:list')")
    public PageResult<MyOrderVO> getConsumptionRecords(@Parameter(description = "用户ID") @PathVariable Long userId,
                                                       @Valid @ParameterObject MyOrderQuery query) {
        return PageResult.success(orderService.listByUserId(userId, query));
    }

    @Operation(summary = "会员权益使用明细（管理端）")
    @GetMapping("/{userId}/benefit-usage")
    @PreAuthorize("@ss.hasPerm('member:list')")
    public Result<Map<String, Object>> getBenefitUsage(@Parameter(description = "用户ID") @PathVariable Long userId) {
        return Result.success(memberService.getBenefitSummary(userId));
    }

    @Operation(summary = "会员操作日志（管理端）")
    @GetMapping("/{userId}/operation-logs")
    @PreAuthorize("@ss.hasPerm('member:list')")
    public Result<Map<String, Object>> getOperationLogs(@Parameter(description = "用户ID") @PathVariable Long userId,
                                                        @Parameter(description = "页码") @RequestParam(defaultValue = "1") Integer pageNum,
                                                        @Parameter(description = "每页条数") @RequestParam(defaultValue = "10") Integer pageSize) {
        return Result.success(memberService.listMemberAuditLogs(userId, pageNum, pageSize));
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
