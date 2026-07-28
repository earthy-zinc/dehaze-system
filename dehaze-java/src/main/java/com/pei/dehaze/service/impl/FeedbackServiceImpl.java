package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysFeedbackMapper;
import com.pei.dehaze.mapper.SysFeedbackReplyMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysFeedback;
import com.pei.dehaze.model.entity.SysFeedbackReply;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.FeedbackAssignForm;
import com.pei.dehaze.model.form.FeedbackCloseForm;
import com.pei.dehaze.model.form.FeedbackCreateForm;
import com.pei.dehaze.model.form.FeedbackReplyForm;
import com.pei.dehaze.model.form.FeedbackSupplementForm;
import com.pei.dehaze.model.query.FeedbackPageQuery;
import com.pei.dehaze.model.vo.FeedbackDetailVO;
import com.pei.dehaze.model.vo.FeedbackPageVO;
import com.pei.dehaze.model.vo.FeedbackReplyVO;
import com.pei.dehaze.model.vo.FeedbackStatsVO;
import com.pei.dehaze.model.vo.IdVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.FeedbackService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class FeedbackServiceImpl extends ServiceImpl<SysFeedbackMapper, SysFeedback> implements FeedbackService {

    private static final int DAILY_FEEDBACK_LIMIT = 5;
    private static final int FEEDBACK_IMAGE_MAX_COUNT = 5;
    private static final int FEEDBACK_STATUS_PENDING = 1;
    private static final int FEEDBACK_STATUS_PROCESSING = 2;
    private static final int FEEDBACK_STATUS_REPLIED = 3;
    private static final int FEEDBACK_STATUS_CLOSED = 4;
    private static final int REPLIER_TYPE_USER = 1;
    private static final int REPLIER_TYPE_ADMIN = 2;

    private static final Map<Integer, String> STATUS_TO_STRING = Map.of(
            1, "pending", 2, "processing", 3, "replied", 4, "closed");
    private static final Map<String, Integer> STATUS_FROM_STRING = Map.of(
            "pending", 1, "processing", 2, "replied", 3, "closed", 4);
    private static final List<String> FEEDBACK_TYPES = Arrays.asList(
            "suggestion", "bug", "experience", "complaint");
    private static final List<String> ALLOWED_IMAGE_EXTENSIONS = Arrays.asList(".jpg", ".jpeg", ".png", ".webp");
    private static final String CACHE_FEEDBACK_STATS = "feedback:stats";

    private final SysFeedbackReplyMapper feedbackReplyMapper;
    private final SysUserMapper userMapper;
    private final ObjectMapper objectMapper;
    private final StringRedisTemplate stringRedisTemplate;

    @Value("${file.baseUrl:}")
    private String fileBaseUrl;

    @Override
    public IdVO createFeedback(FeedbackCreateForm form) {
        validateImageUrls(form.getImages(), FEEDBACK_IMAGE_MAX_COUNT);
        Long userId = SecurityUtils.getUserId();
        String dailyKey = "feedback:daily:" + userId + ":" + LocalDate.now();
        String countStr = stringRedisTemplate.opsForValue().get(dailyKey);
        int todayCount = countStr != null ? Integer.parseInt(countStr) : 0;
        if (todayCount >= DAILY_FEEDBACK_LIMIT) {
            throw new BusinessException(ResultCode.FEEDBACK_LIMIT_EXCEEDED);
        }
        SysFeedback feedback = new SysFeedback();
        feedback.setUserId(userId);
        feedback.setFeedbackType(form.getFeedbackType());
        feedback.setTitle(form.getTitle());
        feedback.setContent(form.getContent());
        feedback.setContact(form.getContact());
        feedback.setImages(serializeList(form.getImages()));
        feedback.setRelatedModule(form.getRelatedModule());
        feedback.setStatus(FEEDBACK_STATUS_PENDING);
        feedback.setPriority(1);
        this.save(feedback);

        Long newCount = stringRedisTemplate.opsForValue().increment(dailyKey);
        if (newCount != null && newCount == 1L) {
            stringRedisTemplate.expire(dailyKey, Duration.ofHours(25));
        }
        try {
            stringRedisTemplate.delete(CACHE_FEEDBACK_STATS);
        } catch (Exception e) {
            log.warn("清除反馈统计缓存失败", e);
        }
        return new IdVO(feedback.getId());
    }

    @Override
    public Page<FeedbackPageVO> listMyFeedback(int pageNum, int pageSize) {
        Long userId = SecurityUtils.getUserId();
        Page<SysFeedback> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<SysFeedback> wrapper = new LambdaQueryWrapper<SysFeedback>()
                .eq(SysFeedback::getUserId, userId)
                .orderByDesc(SysFeedback::getId);
        this.page(page, wrapper);
        Page<FeedbackPageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toPageVO).toList());
        return result;
    }

    @Override
    public FeedbackDetailVO getFeedbackDetail(Long id) {
        SysFeedback feedback = this.getById(id);
        if (feedback == null) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        FeedbackDetailVO vo = new FeedbackDetailVO();
        copyPageFields(feedback, vo);
        vo.setContact(feedback.getContact());
        vo.setImages(parseList(feedback.getImages()));
        vo.setAssignedTime(feedback.getAssignedTime());
        vo.setCloseReason(feedback.getCloseReason());
        List<SysFeedbackReply> replies = feedbackReplyMapper.selectList(
                new LambdaQueryWrapper<SysFeedbackReply>()
                        .eq(SysFeedbackReply::getFeedbackId, id)
                        .orderByAsc(SysFeedbackReply::getId));
        Map<Long, SysUser> userMap = userMapper.selectBatchIds(replies.stream()
                        .map(SysFeedbackReply::getReplierId).distinct().toList())
                .stream().collect(Collectors.toMap(SysUser::getId, u -> u));
        vo.setReplies(replies.stream().map(r -> toReplyVO(r, userMap.get(r.getReplierId()))).toList());
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void supplementFeedback(Long id, FeedbackSupplementForm form) {
        SysFeedback feedback = this.getById(id);
        if (feedback == null) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        if (feedback.getStatus() == FEEDBACK_STATUS_CLOSED) {
            throw new BusinessException(ResultCode.FEEDBACK_CLOSED);
        }
        Long userId = SecurityUtils.getUserId();
        SysFeedbackReply reply = new SysFeedbackReply();
        reply.setFeedbackId(id);
        reply.setReplierId(userId);
        reply.setReplierType(REPLIER_TYPE_USER);
        reply.setContent(form.getContent());
        reply.setReplyType("info");
        reply.setAttachments(serializeList(form.getAttachments()));
        feedbackReplyMapper.insert(reply);

        if (feedback.getStatus() == FEEDBACK_STATUS_REPLIED) {
            feedback.setStatus(FEEDBACK_STATUS_PROCESSING);
            this.updateById(feedback);
        }
    }

    @Override
    public Page<FeedbackPageVO> listPagedFeedback(FeedbackPageQuery query) {
        Page<SysFeedback> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysFeedback> wrapper = new LambdaQueryWrapper<SysFeedback>()
                .eq(CharSequenceUtil.isNotBlank(query.getFeedbackType()),
                        SysFeedback::getFeedbackType, query.getFeedbackType())
                .eq(query.getStatus() != null, SysFeedback::getStatus, statusToInt(query.getStatus()))
                .eq(CharSequenceUtil.isNotBlank(query.getRelatedModule()),
                        SysFeedback::getRelatedModule, query.getRelatedModule())
                .eq(query.getPriority() != null, SysFeedback::getPriority, query.getPriority())
                .eq(query.getAssigneeId() != null, SysFeedback::getAssigneeId, query.getAssigneeId())
                .ge(query.getStartTime() != null, SysFeedback::getCreateTime, query.getStartTime())
                .le(query.getEndTime() != null, SysFeedback::getCreateTime, query.getEndTime())
                .orderByDesc(SysFeedback::getId);

        if (CharSequenceUtil.isNotBlank(query.getKeywords())) {
            List<Long> userIds = userMapper.selectList(new LambdaQueryWrapper<SysUser>()
                            .and(w -> w.like(SysUser::getUsername, query.getKeywords())
                                    .or().like(SysUser::getNickname, query.getKeywords())))
                    .stream().map(SysUser::getId).toList();
            if (userIds.isEmpty()) {
                wrapper.and(w -> w.like(SysFeedback::getTitle, query.getKeywords())
                        .or().like(SysFeedback::getContent, query.getKeywords()));
            } else {
                wrapper.and(w -> w.like(SysFeedback::getTitle, query.getKeywords())
                        .or().like(SysFeedback::getContent, query.getKeywords())
                        .or().in(SysFeedback::getUserId, userIds));
            }
        }

        this.page(page, wrapper);
        List<Long> allUserIds = page.getRecords().stream()
                .flatMap(f -> Arrays.asList(f.getUserId(), f.getAssigneeId()).stream())
                .filter(java.util.Objects::nonNull)
                .distinct().toList();
        Map<Long, SysUser> userMap = allUserIds.isEmpty() ? Collections.emptyMap()
                : userMapper.selectBatchIds(allUserIds).stream()
                .collect(Collectors.toMap(SysUser::getId, u -> u));
        Page<FeedbackPageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(f -> toPageVO(f, userMap)).toList());
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void assignFeedback(Long id, FeedbackAssignForm form) {
        SysFeedback feedback = this.getById(id);
        if (feedback == null) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        if (feedback.getStatus() == FEEDBACK_STATUS_CLOSED) {
            throw new BusinessException(ResultCode.FEEDBACK_CLOSED);
        }
        LambdaUpdateWrapper<SysFeedback> wrapper = new LambdaUpdateWrapper<SysFeedback>()
                .eq(SysFeedback::getId, id)
                .set(SysFeedback::getAssigneeId, form.getAssigneeId())
                .set(SysFeedback::getAssignedTime, LocalDateTime.now());
        if (feedback.getStatus() == FEEDBACK_STATUS_PENDING) {
            wrapper.set(SysFeedback::getStatus, FEEDBACK_STATUS_PROCESSING);
        }
        this.update(wrapper);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void replyFeedback(Long id, FeedbackReplyForm form) {
        SysFeedback feedback = this.getById(id);
        if (feedback == null) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        if (feedback.getStatus() == FEEDBACK_STATUS_CLOSED) {
            throw new BusinessException(ResultCode.FEEDBACK_CLOSED);
        }
        Long adminId = SecurityUtils.getUserId();
        SysFeedbackReply reply = new SysFeedbackReply();
        reply.setFeedbackId(id);
        reply.setReplierId(adminId);
        reply.setReplierType(REPLIER_TYPE_ADMIN);
        reply.setContent(form.getContent());
        reply.setReplyType(form.getReplyType());
        reply.setAttachments(serializeList(form.getAttachments()));
        feedbackReplyMapper.insert(reply);

        LambdaUpdateWrapper<SysFeedback> wrapper = new LambdaUpdateWrapper<SysFeedback>()
                .eq(SysFeedback::getId, id)
                .set(SysFeedback::getStatus, FEEDBACK_STATUS_REPLIED);
        if (feedback.getAssigneeId() == null) {
            wrapper.set(SysFeedback::getAssigneeId, adminId);
            wrapper.set(SysFeedback::getAssignedTime, LocalDateTime.now());
        }
        this.update(wrapper);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void closeFeedback(Long id, FeedbackCloseForm form) {
        SysFeedback feedback = this.getById(id);
        if (feedback == null) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        if (feedback.getStatus() == FEEDBACK_STATUS_CLOSED) {
            throw new BusinessException(ResultCode.FEEDBACK_CLOSED);
        }
        LambdaUpdateWrapper<SysFeedback> wrapper = new LambdaUpdateWrapper<SysFeedback>()
                .eq(SysFeedback::getId, id)
                .set(SysFeedback::getStatus, FEEDBACK_STATUS_CLOSED)
                .set(SysFeedback::getCloseReason, form.getCloseReason());
        this.update(wrapper);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void updateTags(Long id, List<String> tags) {
        SysFeedback feedback = this.getById(id);
        if (feedback == null) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        LambdaUpdateWrapper<SysFeedback> wrapper = new LambdaUpdateWrapper<SysFeedback>()
                .eq(SysFeedback::getId, id)
                .set(SysFeedback::getTags, serializeList(tags));
        this.update(wrapper);
    }

    @Override
    public FeedbackStatsVO getFeedbackStats(LocalDateTime startTime, LocalDateTime endTime) {
        if (startTime == null && endTime == null) {
            try {
                String cached = stringRedisTemplate.opsForValue().get(CACHE_FEEDBACK_STATS);
                if (cached != null) {
                    return objectMapper.readValue(cached, FeedbackStatsVO.class);
                }
            } catch (Exception e) {
                log.warn("读取反馈统计缓存失败", e);
                stringRedisTemplate.delete(CACHE_FEEDBACK_STATS);
            }
            FeedbackStatsVO stats = calcFeedbackStats(startTime, endTime);
            try {
                String json = objectMapper.writeValueAsString(stats);
                stringRedisTemplate.opsForValue().set(CACHE_FEEDBACK_STATS, json, 10, TimeUnit.MINUTES);
            } catch (Exception e) {
                log.warn("写入反馈统计缓存失败", e);
            }
            return stats;
        }
        return calcFeedbackStats(startTime, endTime);
    }

    private FeedbackStatsVO calcFeedbackStats(LocalDateTime startTime, LocalDateTime endTime) {
        LambdaQueryWrapper<SysFeedback> wrapper = new LambdaQueryWrapper<SysFeedback>()
                .ge(startTime != null, SysFeedback::getCreateTime, startTime)
                .le(endTime != null, SysFeedback::getCreateTime, endTime);
        List<SysFeedback> feedbacks = this.list(wrapper);

        FeedbackStatsVO stats = new FeedbackStatsVO();
        stats.setTotalFeedback((long) feedbacks.size());

        Map<String, Long> typeDist = new LinkedHashMap<>();
        for (String t : FEEDBACK_TYPES) {
            typeDist.put(t, 0L);
        }
        for (SysFeedback f : feedbacks) {
            if (f.getFeedbackType() != null && typeDist.containsKey(f.getFeedbackType())) {
                typeDist.merge(f.getFeedbackType(), 1L, Long::sum);
            }
        }
        stats.setTypeDistribution(typeDist);

        Map<String, Long> statusDist = new LinkedHashMap<>();
        for (String s : Arrays.asList("pending", "processing", "replied", "closed")) {
            statusDist.put(s, 0L);
        }
        for (SysFeedback f : feedbacks) {
            String status = STATUS_TO_STRING.get(f.getStatus());
            if (status != null) {
                statusDist.merge(status, 1L, Long::sum);
            }
        }
        stats.setStatusDistribution(statusDist);

        Map<String, Long> moduleCount = new LinkedHashMap<>();
        for (SysFeedback f : feedbacks) {
            if (CharSequenceUtil.isNotBlank(f.getRelatedModule())) {
                moduleCount.merge(f.getRelatedModule(), 1L, Long::sum);
            }
        }
        stats.setModuleDistribution(moduleCount.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .map(e -> {
                    FeedbackStatsVO.ModuleCount mc = new FeedbackStatsVO.ModuleCount();
                    mc.setModule(e.getKey());
                    mc.setCount(e.getValue());
                    return mc;
                })
                .toList());

        List<Long> feedbackIds = feedbacks.stream().map(SysFeedback::getId).toList();
        if (!feedbackIds.isEmpty()) {
            List<SysFeedbackReply> allReplies = feedbackReplyMapper.selectList(
                    new LambdaQueryWrapper<SysFeedbackReply>()
                            .in(SysFeedbackReply::getFeedbackId, feedbackIds)
                            .orderByAsc(SysFeedbackReply::getFeedbackId)
                            .orderByAsc(SysFeedbackReply::getId));
            Map<Long, List<SysFeedbackReply>> replyMap = allReplies.stream()
                    .collect(Collectors.groupingBy(SysFeedbackReply::getFeedbackId));

            long totalResponseTime = 0;
            long responseCount = 0;
            long totalCloseTime = 0;
            long closeCount = 0;
            for (SysFeedback f : feedbacks) {
                List<SysFeedbackReply> replies = replyMap.get(f.getId());
                if (replies != null && !replies.isEmpty()) {
                    SysFeedbackReply firstReply = replies.get(0);
                    if (f.getCreateTime() != null && firstReply.getCreateTime() != null) {
                        totalResponseTime += java.time.Duration.between(
                                f.getCreateTime(), firstReply.getCreateTime()).toMillis();
                        responseCount++;
                    }
                }
                if (f.getStatus() != null && f.getStatus() == FEEDBACK_STATUS_CLOSED
                        && f.getCreateTime() != null && f.getUpdateTime() != null) {
                    totalCloseTime += java.time.Duration.between(
                            f.getCreateTime(), f.getUpdateTime()).toMillis();
                    closeCount++;
                }
            }
            stats.setAverageResponseTime(responseCount > 0 ? totalResponseTime / responseCount : 0L);
            stats.setAverageCloseTime(closeCount > 0 ? totalCloseTime / closeCount : 0L);
        } else {
            stats.setAverageResponseTime(0L);
            stats.setAverageCloseTime(0L);
        }
        stats.setTopKeywords(Collections.emptyList());
        return stats;
    }

    private Integer statusToInt(String status) {
        if (CharSequenceUtil.isBlank(status)) {
            return null;
        }
        return STATUS_FROM_STRING.get(status);
    }

    private void copyPageFields(SysFeedback feedback, FeedbackPageVO vo) {
        vo.setId(feedback.getId());
        vo.setUserId(feedback.getUserId());
        vo.setFeedbackType(feedback.getFeedbackType());
        vo.setTitle(feedback.getTitle());
        vo.setContent(feedback.getContent());
        vo.setStatus(STATUS_TO_STRING.get(feedback.getStatus()));
        vo.setPriority(feedback.getPriority());
        vo.setAssigneeId(feedback.getAssigneeId());
        vo.setRelatedModule(feedback.getRelatedModule());
        List<String> tags = parseList(feedback.getTags());
        vo.setTags(tags != null ? tags : Collections.emptyList());
        vo.setCreateTime(feedback.getCreateTime());
        vo.setUpdateTime(feedback.getUpdateTime());
    }

    private FeedbackPageVO toPageVO(SysFeedback feedback) {
        return toPageVO(feedback, Collections.emptyMap());
    }

    private FeedbackPageVO toPageVO(SysFeedback feedback, Map<Long, SysUser> userMap) {
        FeedbackPageVO vo = new FeedbackPageVO();
        copyPageFields(feedback, vo);
        SysUser user = userMap.get(feedback.getUserId());
        if (user != null) {
            vo.setUsername(user.getUsername());
        }
        if (feedback.getAssigneeId() != null) {
            SysUser assignee = userMap.get(feedback.getAssigneeId());
            if (assignee != null) {
                vo.setAssigneeName(assignee.getUsername());
            }
        }
        return vo;
    }

    private FeedbackReplyVO toReplyVO(SysFeedbackReply reply, SysUser user) {
        FeedbackReplyVO vo = new FeedbackReplyVO();
        vo.setId(reply.getId());
        vo.setFeedbackId(reply.getFeedbackId());
        vo.setReplierId(reply.getReplierId());
        vo.setReplierName(user != null ? user.getUsername() : null);
        vo.setReplierType(reply.getReplierType());
        vo.setContent(reply.getContent());
        vo.setReplyType(reply.getReplyType());
        vo.setAttachments(parseList(reply.getAttachments()));
        vo.setCreateTime(reply.getCreateTime());
        return vo;
    }

    private List<String> parseList(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, new TypeReference<List<String>>() {});
        } catch (JsonProcessingException e) {
            log.warn("解析JSON List失败: {}", json, e);
            return null;
        }
    }

    private String serializeList(List<String> list) {
        if (list == null || list.isEmpty()) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(list);
        } catch (JsonProcessingException e) {
            log.warn("序列化List到JSON失败", e);
            return null;
        }
    }

    private void validateImageUrls(List<String> urls, int maxCount) {
        if (urls == null || urls.isEmpty()) {
            return;
        }
        if (urls.size() > maxCount) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "图片数量不能超过" + maxCount + "张");
        }
        for (String url : urls) {
            if (CharSequenceUtil.isBlank(url)) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "图片URL不能为空");
            }
            if (CharSequenceUtil.isNotBlank(fileBaseUrl) && !url.startsWith(fileBaseUrl)) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "图片URL必须为MinIO域名");
            }
            String path = url.split("[?#]")[0].toLowerCase();
            boolean validExt = ALLOWED_IMAGE_EXTENSIONS.stream().anyMatch(path::endsWith);
            if (!validExt) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "图片格式不支持，仅支持jpg/jpeg/png/webp");
            }
        }
    }
}
