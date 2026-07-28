package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.FeedbackAssignForm;
import com.pei.dehaze.model.form.FeedbackCloseForm;
import com.pei.dehaze.model.form.FeedbackCreateForm;
import com.pei.dehaze.model.form.FeedbackReplyForm;
import com.pei.dehaze.model.form.FeedbackSupplementForm;
import com.pei.dehaze.model.form.RatingCreateForm;
import com.pei.dehaze.model.form.RatingReplyForm;
import com.pei.dehaze.model.query.FeedbackPageQuery;
import com.pei.dehaze.model.query.RatingPageQuery;
import com.pei.dehaze.model.vo.FeedbackDetailVO;
import com.pei.dehaze.model.vo.FeedbackPageVO;
import com.pei.dehaze.model.vo.FeedbackStatsVO;
import com.pei.dehaze.model.vo.IdVO;
import com.pei.dehaze.model.vo.MyRatingVO;
import com.pei.dehaze.model.vo.RatingDetailVO;
import com.pei.dehaze.model.vo.RatingPageVO;
import com.pei.dehaze.model.vo.RatingStatsVO;
import com.pei.dehaze.service.FeedbackService;
import com.pei.dehaze.service.RatingService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@Tag(name = "14.反馈评价")
@RestController
@RequestMapping("/api/v1/feedback")
@RequiredArgsConstructor
public class FeedbackController {

    private final RatingService ratingService;
    private final FeedbackService feedbackService;

    // ============ 评价接口 - 用户端 ============

    @Operation(summary = "用户端：提交评分")
    @PostMapping("/ratings")
    public Result<IdVO> createRating(@Valid @RequestBody RatingCreateForm form) {
        return Result.success(ratingService.createRating(form));
    }

    @Operation(summary = "用户端：修改评分")
    @PutMapping("/ratings/{id}")
    public Result<Void> updateRating(@Parameter(description = "评价ID") @PathVariable Long id,
                                     @Valid @RequestBody RatingCreateForm form) {
        ratingService.updateRating(id, form);
        return Result.success();
    }

    @Operation(summary = "用户端：我的评价列表")
    @GetMapping("/ratings/my")
    public PageResult<MyRatingVO> listMyRatings(
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") int pageNum,
            @Parameter(description = "每页记录数") @RequestParam(defaultValue = "10") int pageSize) {
        Page<MyRatingVO> page = ratingService.listMyRatings(pageNum, pageSize);
        return PageResult.success(page);
    }

    @Operation(summary = "用户端：按处理记录查评价")
    @GetMapping("/ratings/by-prediction/{predictionLogId}")
    public Result<RatingDetailVO> getRatingByPrediction(
            @Parameter(description = "处理记录ID") @PathVariable Long predictionLogId) {
        return Result.success(ratingService.getRatingByPrediction(predictionLogId));
    }

    // ============ 评价接口 - 后台 ============

    @Operation(summary = "后台：评价分页列表")
    @GetMapping("/ratings/page")
    public PageResult<RatingPageVO> listRatings(@ParameterObject RatingPageQuery query) {
        Page<RatingPageVO> page = ratingService.listPagedRatings(query);
        return PageResult.success(page);
    }

    @Operation(summary = "后台：隐藏评价")
    @PutMapping("/ratings/{id}/hide")
    @PreAuthorize("@ss.hasPerm('feedback:rating:edit')")
    public Result<Void> hideRating(@Parameter(description = "评价ID") @PathVariable Long id) {
        ratingService.hideRating(id);
        return Result.success();
    }

    @Operation(summary = "后台：回复评价")
    @PostMapping("/ratings/{id}/reply")
    @PreAuthorize("@ss.hasPerm('feedback:rating:reply')")
    public Result<Void> replyRating(@Parameter(description = "评价ID") @PathVariable Long id,
                                    @Valid @RequestBody RatingReplyForm form) {
        ratingService.replyRating(id, form.getContent());
        return Result.success();
    }

    @Operation(summary = "后台：评价统计")
    @GetMapping("/ratings/stats")
    public Result<RatingStatsVO> getRatingStats(
            @Parameter(description = "开始时间(yyyy-MM-dd HH:mm:ss)") @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @Parameter(description = "结束时间(yyyy-MM-dd HH:mm:ss)") @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime) {
        return Result.success(ratingService.getRatingStats(startTime, endTime));
    }

    // ============ 反馈接口 - 用户端 ============

    @Operation(summary = "用户端：提交反馈")
    @PostMapping
    public Result<IdVO> createFeedback(@Valid @RequestBody FeedbackCreateForm form) {
        return Result.success(feedbackService.createFeedback(form));
    }

    @Operation(summary = "用户端：我的反馈列表")
    @GetMapping("/my")
    public PageResult<FeedbackPageVO> listMyFeedback(
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") int pageNum,
            @Parameter(description = "每页记录数") @RequestParam(defaultValue = "10") int pageSize) {
        Page<FeedbackPageVO> page = feedbackService.listMyFeedback(pageNum, pageSize);
        return PageResult.success(page);
    }

    @Operation(summary = "用户端/后台：反馈详情")
    @GetMapping("/{id}")
    public Result<FeedbackDetailVO> getFeedbackDetail(@Parameter(description = "反馈ID") @PathVariable Long id) {
        return Result.success(feedbackService.getFeedbackDetail(id));
    }

    @Operation(summary = "用户端：补充说明")
    @PostMapping("/{id}/supplement")
    public Result<Void> supplementFeedback(@Parameter(description = "反馈ID") @PathVariable Long id,
                                          @Valid @RequestBody FeedbackSupplementForm form) {
        feedbackService.supplementFeedback(id, form);
        return Result.success();
    }

    // ============ 反馈接口 - 后台 ============

    @Operation(summary = "后台：反馈分页列表")
    @GetMapping("/page")
    public PageResult<FeedbackPageVO> listFeedback(@ParameterObject FeedbackPageQuery query) {
        Page<FeedbackPageVO> page = feedbackService.listPagedFeedback(query);
        return PageResult.success(page);
    }

    @Operation(summary = "后台：分配处理人")
    @PutMapping("/{id}/assign")
    @PreAuthorize("@ss.hasPerm('feedback:assign')")
    public Result<Void> assignFeedback(@Parameter(description = "反馈ID") @PathVariable Long id,
                                       @Valid @RequestBody FeedbackAssignForm form) {
        feedbackService.assignFeedback(id, form);
        return Result.success();
    }

    @Operation(summary = "后台：回复反馈")
    @PostMapping("/{id}/reply")
    @PreAuthorize("@ss.hasPerm('feedback:reply')")
    public Result<Void> replyFeedback(@Parameter(description = "反馈ID") @PathVariable Long id,
                                      @Valid @RequestBody FeedbackReplyForm form) {
        feedbackService.replyFeedback(id, form);
        return Result.success();
    }

    @Operation(summary = "后台：关闭反馈")
    @PutMapping("/{id}/close")
    @PreAuthorize("@ss.hasPerm('feedback:close')")
    public Result<Void> closeFeedback(@Parameter(description = "反馈ID") @PathVariable Long id,
                                     @Valid @RequestBody FeedbackCloseForm form) {
        feedbackService.closeFeedback(id, form);
        return Result.success();
    }

    @Operation(summary = "后台：设置反馈标签")
    @PutMapping("/{id}/tags")
    @PreAuthorize("@ss.hasPerm('feedback:edit')")
    public Result<Void> updateFeedbackTags(@Parameter(description = "反馈ID") @PathVariable Long id,
                                          @RequestBody List<String> tags) {
        feedbackService.updateTags(id, tags);
        return Result.success();
    }

    @Operation(summary = "后台：反馈统计")
    @GetMapping("/stats")
    public Result<FeedbackStatsVO> getFeedbackStats(
            @Parameter(description = "开始时间(yyyy-MM-dd HH:mm:ss)") @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @Parameter(description = "结束时间(yyyy-MM-dd HH:mm:ss)") @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime) {
        return Result.success(feedbackService.getFeedbackStats(startTime, endTime));
    }
}
