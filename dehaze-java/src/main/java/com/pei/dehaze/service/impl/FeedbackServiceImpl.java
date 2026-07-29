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
    @Transactional(rollbackFor = Exception.class)
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
        Long userId = SecurityUtils.getUserId();
        boolean isAdmin = SecurityUtils.isAdmin();
        SysFeedback feedback = this.getById(id);
        if (feedback == null) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        if (!isAdmin && !feedback.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        FeedbackDetailVO vo = new FeedbackDetailVO();
        copyPageFields(feedback, vo);
        if (isAdmin) {
            vo.setContact(feedback.getContact());
        }
        vo.setImages(parseList(feedback.getImages()));
        vo.setAssignedTime(feedback.getAssignedTime());
        vo.setCloseReason(feedback.getCloseReason());
        List<SysFeedbackReply> replies = feedbackReplyMapper.selectList(
                new LambdaQueryWrapper<SysFeedbackReply>()
                        .eq(SysFeedbackReply::getFeedbackId, id)
                        .orderByAsc(SysFeedbackReply::getId));
        Map<Long, SysUser> userMap = replies.isEmpty() ? Collections.emptyMap()
                : userMapper.selectBatchIds(replies.stream()
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
        Long userId = SecurityUtils.getUserId();
        if (!feedback.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND);
        }
        if (feedback.getStatus() == FEEDBACK_STATUS_CLOSED) {
            throw new BusinessException(ResultCode.FEEDBACK_CLOSED);
        }
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
        FeedbackStatsVO stats = new FeedbackStatsVO();
        stats.setTotalFeedback(baseMapper.countTotal(startTime, endTime));

        Map<String, Long> typeDist = new LinkedHashMap<>();
        for (String t : FEEDBACK_TYPES) {
            typeDist.put(t, 0L);
        }
        for (Map<String, Object> row : baseMapper.selectTypeDistribution(startTime, endTime)) {
            String type = (String) row.get("feedbackType");
            Long count = ((Number) row.get("cnt")).longValue();
            if (type != null && typeDist.containsKey(type)) {
                typeDist.put(type, count);
            }
        }
        stats.setTypeDistribution(typeDist);

        Map<String, Long> statusDist = new LinkedHashMap<>();
        for (String s : Arrays.asList("pending", "processing", "replied", "closed")) {
            statusDist.put(s, 0L);
        }
        for (Map<String, Object> row : baseMapper.selectStatusDistribution(startTime, endTime)) {
            Integer statusVal = ((Number) row.get("status")).intValue();
            Long count = ((Number) row.get("cnt")).longValue();
            String status = STATUS_TO_STRING.get(statusVal);
            if (status != null) {
                statusDist.put(status, count);
            }
        }
        stats.setStatusDistribution(statusDist);

        Map<String, Long> moduleCount = new LinkedHashMap<>();
        for (Map<String, Object> row : baseMapper.selectModuleDistribution(startTime, endTime)) {
            String module = (String) row.get("relatedModule");
            Long count = ((Number) row.get("cnt")).longValue();
            moduleCount.put(module, count);
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

        Map<Long, java.time.LocalDateTime> firstReplyMap = new HashMap<>();
        for (Map<String, Object> row : baseMapper.selectFirstReplyTimes(startTime, endTime)) {
            Number feedbackId = (Number) row.get("feedbackId");
            java.time.LocalDateTime firstReplyTime = (java.time.LocalDateTime) row.get("firstReplyTime");
            if (feedbackId != null && firstReplyTime != null) {
                firstReplyMap.put(feedbackId.longValue(), firstReplyTime);
            }
        }

        long totalResponseTime = 0;
        long responseCount = 0;
        long totalCloseTime = 0;
        long closeCount = 0;
        for (Map<String, Object> row : baseMapper.selectFeedbackTimes(startTime, endTime)) {
            Long feedbackId = ((Number) row.get("id")).longValue();
            java.time.LocalDateTime createTime = (java.time.LocalDateTime) row.get("createTime");
            java.time.LocalDateTime updateTime = (java.time.LocalDateTime) row.get("updateTime");
            java.time.LocalDateTime firstReplyTime = firstReplyMap.get(feedbackId);
            if (createTime != null && firstReplyTime != null) {
                totalResponseTime += java.time.Duration.between(createTime, firstReplyTime).toMillis();
                responseCount++;
            }
            Integer statusVal = ((Number) row.get("status")).intValue();
            if (statusVal != null && statusVal == FEEDBACK_STATUS_CLOSED
                    && createTime != null && updateTime != null) {
                totalCloseTime += java.time.Duration.between(createTime, updateTime).toMillis();
                closeCount++;
            }
        }
        stats.setAverageResponseTime(responseCount > 0 ? totalResponseTime / responseCount : 0L);
        stats.setAverageCloseTime(closeCount > 0 ? totalCloseTime / closeCount : 0L);

        stats.setTopKeywords(topKeywords(startTime, endTime, 10));
        return stats;
    }

    private List<FeedbackStatsVO.KeywordCount> topKeywords(LocalDateTime startTime, LocalDateTime endTime, int limit) {
        Map<String, Long> counts = new LinkedHashMap<>();
        String separators = " \t\n\r,，。.!！?？;；:：、\"'()（）[]【】{}/\\|";
        for (Map<String, Object> row : baseMapper.selectTitleAndContent(startTime, endTime)) {
            String title = (String) row.get("title");
            String content = (String) row.get("content");
            String text = (title != null ? title : "") + " " + (content != null ? content : "");
            java.util.StringTokenizer tokenizer = new java.util.StringTokenizer(text, separators);
            while (tokenizer.hasMoreTokens()) {
                String word = tokenizer.nextToken();
                if (word.length() >= 2) {
                    counts.merge(word, 1L, Long::sum);
                }
            }
        }
        return counts.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .limit(limit)
                .map(e -> {
                    FeedbackStatsVO.KeywordCount kc = new FeedbackStatsVO.KeywordCount();
                    kc.setKeyword(e.getKey());
                    kc.setCount(e.getValue());
                    return kc;
                })
                .toList();
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
