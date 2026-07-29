package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.filter.TraceIdFilter;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.mapper.SysRatingMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.entity.SysRating;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.MemberGrowthAdjustForm;
import com.pei.dehaze.model.form.RatingCreateForm;
import com.pei.dehaze.model.query.RatingPageQuery;
import com.pei.dehaze.model.vo.IdVO;
import com.pei.dehaze.model.vo.MyRatingVO;
import com.pei.dehaze.model.vo.RatingDetailVO;
import com.pei.dehaze.model.vo.RatingPageVO;
import com.pei.dehaze.model.vo.RatingStatsVO;
import com.pei.dehaze.mq.RabbitMQPublisher;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.MemberService;
import com.pei.dehaze.service.RatingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class RatingServiceImpl extends ServiceImpl<SysRatingMapper, SysRating> implements RatingService {

    private static final List<String> POSITIVE_TAGS = Arrays.asList(
            "去雾彻底", "色彩自然", "细节清晰", "处理速度快", "整体提升明显");
    private static final List<String> NEGATIVE_TAGS = Arrays.asList(
            "残留雾气", "色彩失真", "细节丢失", "处理速度慢", "无明显改善");

    private static final int RATING_IMAGE_MAX_COUNT = 3;
    private static final int RATING_GROWTH_VALUE = 5;
    private static final int RATING_DAILY_GROWTH_LIMIT = 5;
    private static final List<String> ALLOWED_IMAGE_EXTENSIONS = Arrays.asList(".jpg", ".jpeg", ".png", ".webp");
    private static final String CACHE_RATING_STATS_GLOBAL = "rating:stats:global";

    private final SysPredLogMapper predLogMapper;
    private final SysAlgorithmMapper algorithmMapper;
    private final SysUserMapper userMapper;
    private final ObjectMapper objectMapper;
    private final StringRedisTemplate stringRedisTemplate;
    private final MemberService memberService;
    private final ObjectProvider<RabbitMQPublisher> rabbitMQPublisherProvider;

    @Value("${file.baseUrl:}")
    private String fileBaseUrl;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public IdVO createRating(RatingCreateForm form) {
        validateImageUrls(form.getImageUrls(), RATING_IMAGE_MAX_COUNT);
        Long userId = SecurityUtils.getUserId();
        SysPredLog predLog = predLogMapper.selectById(form.getPredLogId());
        if (predLog == null) {
            throw new BusinessException(ResultCode.PREDICTION_LOG_NOT_FOUND);
        }
        if (predLog.getStatus() != LogStatusEnum.COMPLETED) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW);
        }
        if (!userId.equals(predLog.getCreateBy())) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW);
        }
        Long existCount = this.count(new LambdaQueryWrapper<SysRating>()
                .eq(SysRating::getPredLogId, form.getPredLogId()));
        if (existCount > 0) {
            throw new BusinessException(ResultCode.RATING_ALREADY_EXISTS);
        }
        if (predLog.getUpdateTime() != null
                && predLog.getUpdateTime().isBefore(LocalDateTime.now().minusDays(30))) {
            throw new BusinessException(ResultCode.RATING_EXPIRED);
        }
        SysRating rating = new SysRating();
        rating.setUserId(userId);
        rating.setPredLogId(form.getPredLogId());
        rating.setAlgorithmId(predLog.getAlgorithmId());
        rating.setRating(form.getRating());
        rating.setComment(form.getComment());
        rating.setTags(serializeList(form.getTags()));
        rating.setImageUrls(serializeList(form.getImageUrls()));
        rating.setIsAnonymous(form.getIsAnonymous() != null ? form.getIsAnonymous() : 0);
        rating.setIsHidden(0);
        this.save(rating);

        invalidateRatingCaches(predLog.getAlgorithmId());
        tryGrantGrowth(userId);
        if (rating.getRating() != null && rating.getRating() <= 2) {
            publishLowRatingAlert(rating.getId());
        }
        return new IdVO(rating.getId());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void updateRating(Long id, RatingCreateForm form) {
        validateImageUrls(form.getImageUrls(), RATING_IMAGE_MAX_COUNT);
        Long userId = SecurityUtils.getUserId();
        SysRating rating = this.getById(id);
        if (rating == null) {
            throw new BusinessException(ResultCode.RATING_NOT_FOUND);
        }
        if (!rating.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.RATING_NOT_FOUND);
        }
        rating.setRating(form.getRating());
        rating.setComment(form.getComment());
        rating.setTags(serializeList(form.getTags()));
        rating.setImageUrls(serializeList(form.getImageUrls()));
        if (form.getIsAnonymous() != null) {
            rating.setIsAnonymous(form.getIsAnonymous());
        }
        this.updateById(rating);
    }

    @Override
    public RatingDetailVO getRatingByPrediction(Long predLogId) {
        Long userId = SecurityUtils.getUserId();
        SysPredLog predLog = predLogMapper.selectById(predLogId);
        if (predLog == null) {
            throw new BusinessException(ResultCode.PREDICTION_LOG_NOT_FOUND);
        }
        if (!userId.equals(predLog.getCreateBy())) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW);
        }
        SysRating rating = this.getOne(new LambdaQueryWrapper<SysRating>()
                .eq(SysRating::getPredLogId, predLogId)
                .last("LIMIT 1"));
        if (rating == null) {
            return null;
        }
        SysAlgorithm algorithm = algorithmMapper.selectById(rating.getAlgorithmId());
        return toDetailVO(rating, algorithm, null);
    }

    @Override
    public Page<MyRatingVO> listMyRatings(int pageNum, int pageSize) {
        Long userId = SecurityUtils.getUserId();
        Page<SysRating> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<SysRating> wrapper = new LambdaQueryWrapper<SysRating>()
                .eq(SysRating::getUserId, userId)
                .eq(SysRating::getIsHidden, 0)
                .orderByDesc(SysRating::getId);
        this.page(page, wrapper);
        Map<Long, SysAlgorithm> algorithmMap = loadAlgorithmMap(page.getRecords());
        Page<MyRatingVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream()
                .map(r -> toMyVO(r, algorithmMap.get(r.getAlgorithmId())))
                .toList());
        return result;
    }

    @Override
    public Page<RatingPageVO> listPagedRatings(RatingPageQuery query) {
        Page<SysRating> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysRating> wrapper = new LambdaQueryWrapper<SysRating>()
                .ge(query.getRatingMin() != null, SysRating::getRating, query.getRatingMin())
                .le(query.getRatingMax() != null, SysRating::getRating, query.getRatingMax())
                .eq(query.getAlgorithmId() != null, SysRating::getAlgorithmId, query.getAlgorithmId())
                .ge(query.getStartTime() != null, SysRating::getCreateTime, query.getStartTime())
                .le(query.getEndTime() != null, SysRating::getCreateTime, query.getEndTime())
                .orderByDesc(SysRating::getId);

        if (Boolean.TRUE.equals(query.getHasComment())) {
            wrapper.isNotNull(SysRating::getComment).ne(SysRating::getComment, "");
        } else if (Boolean.FALSE.equals(query.getHasComment())) {
            andEmptyComment(wrapper);
        }

        if (CharSequenceUtil.isNotBlank(query.getKeywords())) {
            List<Long> userIds = userMapper.selectList(new LambdaQueryWrapper<SysUser>()
                            .and(w -> w.like(SysUser::getUsername, query.getKeywords())
                                    .or().like(SysUser::getNickname, query.getKeywords())))
                    .stream().map(SysUser::getId).toList();
            if (userIds.isEmpty()) {
                Page<RatingPageVO> empty = new Page<>(page.getCurrent(), page.getSize(), 0);
                empty.setRecords(Collections.emptyList());
                return empty;
            }
            wrapper.in(SysRating::getUserId, userIds);
        }

        if (query.getTags() != null && !query.getTags().isEmpty()) {
            for (String tag : query.getTags()) {
                wrapper.like(SysRating::getTags, tag);
            }
        }

        this.page(page, wrapper);
        List<SysRating> records = page.getRecords();
        if (records.isEmpty()) {
            Page<RatingPageVO> empty = new Page<>(page.getCurrent(), page.getSize(), 0);
            empty.setRecords(Collections.emptyList());
            return empty;
        }
        Map<Long, SysAlgorithm> algorithmMap = loadAlgorithmMap(records);
        Map<Long, SysUser> userMap = userMapper.selectBatchIds(records.stream()
                        .map(SysRating::getUserId).distinct().toList())
                .stream().collect(Collectors.toMap(SysUser::getId, u -> u));
        Page<RatingPageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(records.stream()
                .map(r -> toPageVO(r, algorithmMap.get(r.getAlgorithmId()),
                        userMap.get(r.getUserId())))
                .toList());
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void hideRating(Long id) {
        SysRating rating = this.getById(id);
        if (rating == null) {
            throw new BusinessException(ResultCode.RATING_NOT_FOUND);
        }
        LambdaUpdateWrapper<SysRating> wrapper = new LambdaUpdateWrapper<SysRating>()
                .eq(SysRating::getId, id)
                .set(SysRating::getIsHidden, 1);
        this.update(wrapper);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void replyRating(Long id, String content) {
        SysRating rating = this.getById(id);
        if (rating == null) {
            throw new BusinessException(ResultCode.RATING_NOT_FOUND);
        }
        LambdaUpdateWrapper<SysRating> wrapper = new LambdaUpdateWrapper<SysRating>()
                .eq(SysRating::getId, id)
                .set(SysRating::getAdminReply, content)
                .set(SysRating::getReplyTime, LocalDateTime.now());
        this.update(wrapper);
    }

    @Override
    public RatingStatsVO getRatingStats(LocalDateTime startTime, LocalDateTime endTime) {
        if (startTime == null && endTime == null) {
            try {
                String cached = stringRedisTemplate.opsForValue().get(CACHE_RATING_STATS_GLOBAL);
                if (cached != null) {
                    return objectMapper.readValue(cached, RatingStatsVO.class);
                }
            } catch (Exception e) {
                log.warn("读取评价统计缓存失败", e);
                stringRedisTemplate.delete(CACHE_RATING_STATS_GLOBAL);
            }
            RatingStatsVO stats = calcRatingStats(startTime, endTime);
            try {
                String json = objectMapper.writeValueAsString(stats);
                stringRedisTemplate.opsForValue().set(CACHE_RATING_STATS_GLOBAL, json, 10, TimeUnit.MINUTES);
            } catch (Exception e) {
                log.warn("写入评价统计缓存失败", e);
            }
            return stats;
        }
        return calcRatingStats(startTime, endTime);
    }

    private RatingStatsVO calcRatingStats(LocalDateTime startTime, LocalDateTime endTime) {
        RatingStatsVO stats = new RatingStatsVO();

        Map<Integer, Long> distribution = new LinkedHashMap<>();
        for (int i = 1; i <= 5; i++) {
            distribution.put(i, 0L);
        }
        long totalRatings = 0;
        long ratingSum = 0;
        for (Map<String, Object> row : baseMapper.selectRatingDistribution(startTime, endTime)) {
            Integer rating = ((Number) row.get("rating")).intValue();
            Long count = ((Number) row.get("cnt")).longValue();
            distribution.put(rating, count);
            totalRatings += count;
            ratingSum += (long) rating * count;
        }
        stats.setTotalRatings(totalRatings);
        double avg = totalRatings > 0 ? (double) ratingSum / totalRatings : 0;
        stats.setAverageRating(Math.round(avg * 10) / 10.0);
        stats.setRatingDistribution(distribution);

        Map<String, Long> positiveTagCount = new LinkedHashMap<>();
        Map<String, Long> negativeTagCount = new LinkedHashMap<>();
        for (String tagsJson : baseMapper.selectAllTags(startTime, endTime)) {
            List<String> tags = parseList(tagsJson);
            if (tags == null) continue;
            for (String tag : tags) {
                if (POSITIVE_TAGS.contains(tag)) {
                    positiveTagCount.merge(tag, 1L, Long::sum);
                } else if (NEGATIVE_TAGS.contains(tag)) {
                    negativeTagCount.merge(tag, 1L, Long::sum);
                }
            }
        }
        stats.setPositiveTagRanking(toTagCountList(positiveTagCount));
        stats.setNegativeTagRanking(toTagCountList(negativeTagCount));

        List<RatingStatsVO.AlgorithmStat> algorithmStats = new java.util.ArrayList<>();
        for (Map<String, Object> row : baseMapper.selectAlgorithmStats(startTime, endTime)) {
            RatingStatsVO.AlgorithmStat stat = new RatingStatsVO.AlgorithmStat();
            Number algorithmId = (Number) row.get("algorithmId");
            stat.setAlgorithmId(algorithmId != null ? algorithmId.longValue() : null);
            stat.setAlgorithmName((String) row.get("algorithmName"));
            Number avgRating = (Number) row.get("avgRating");
            stat.setAverageRating(avgRating != null ? Math.round(avgRating.doubleValue() * 10) / 10.0 : 0.0);
            Number total = (Number) row.get("total");
            long totalLong = total != null ? total.longValue() : 0L;
            stat.setTotalRatings(totalLong);
            Number lowCount = (Number) row.get("lowCount");
            long lowLong = lowCount != null ? lowCount.longValue() : 0L;
            stat.setLowRatingRate(totalLong > 0 ? Math.round(lowLong * 10000.0 / totalLong) / 100.0 : 0.0);
            algorithmStats.add(stat);
        }
        stats.setAlgorithmStats(algorithmStats);
        return stats;
    }

    private void andEmptyComment(LambdaQueryWrapper<SysRating> wrapper) {
        wrapper.and(w -> w.isNull(SysRating::getComment).or().eq(SysRating::getComment, ""));
    }

    private Map<Long, SysAlgorithm> loadAlgorithmMap(List<SysRating> records) {
        if (records.isEmpty()) {
            return Collections.emptyMap();
        }
        List<Long> algorithmIds = records.stream()
                .map(SysRating::getAlgorithmId).distinct().toList();
        return algorithmMapper.selectBatchIds(algorithmIds).stream()
                .collect(Collectors.toMap(SysAlgorithm::getId, a -> a));
    }

    private List<RatingStatsVO.TagCount> toTagCountList(Map<String, Long> map) {
        return map.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .limit(5)
                .map(e -> {
                    RatingStatsVO.TagCount tc = new RatingStatsVO.TagCount();
                    tc.setTag(e.getKey());
                    tc.setCount(e.getValue());
                    return tc;
                })
                .toList();
    }

    private MyRatingVO toMyVO(SysRating rating, SysAlgorithm algorithm) {
        MyRatingVO vo = new MyRatingVO();
        vo.setId(rating.getId());
        vo.setPredLogId(rating.getPredLogId());
        vo.setAlgorithmName(algorithm != null ? algorithm.getName() : null);
        vo.setRating(rating.getRating());
        vo.setComment(rating.getComment());
        vo.setTags(parseList(rating.getTags()));
        vo.setImageUrls(parseList(rating.getImageUrls()));
        vo.setIsAnonymous(rating.getIsAnonymous());
        vo.setAdminReply(rating.getAdminReply());
        vo.setReplyTime(rating.getReplyTime());
        vo.setCreateTime(rating.getCreateTime());
        return vo;
    }

    private RatingPageVO toPageVO(SysRating rating, SysAlgorithm algorithm, SysUser user) {
        RatingPageVO vo = new RatingPageVO();
        vo.setId(rating.getId());
        vo.setPredLogId(rating.getPredLogId());
        vo.setAlgorithmName(algorithm != null ? algorithm.getName() : null);
        vo.setRating(rating.getRating());
        vo.setComment(rating.getComment());
        vo.setTags(parseList(rating.getTags()));
        vo.setImageUrls(parseList(rating.getImageUrls()));
        vo.setIsAnonymous(rating.getIsAnonymous());
        vo.setAdminReply(rating.getAdminReply());
        vo.setReplyTime(rating.getReplyTime());
        vo.setCreateTime(rating.getCreateTime());
        vo.setUserId(rating.getUserId());
        if (rating.getIsAnonymous() != null && rating.getIsAnonymous() == 1) {
            vo.setUsername(null);
            vo.setUserAvatar(null);
        } else {
            vo.setUsername(user != null ? user.getUsername() : null);
            vo.setUserAvatar(user != null ? user.getAvatar() : null);
        }
        vo.setIsHidden(rating.getIsHidden());
        return vo;
    }

    private RatingDetailVO toDetailVO(SysRating rating, SysAlgorithm algorithm, SysUser user) {
        RatingDetailVO vo = new RatingDetailVO();
        vo.setId(rating.getId());
        vo.setPredLogId(rating.getPredLogId());
        vo.setAlgorithmName(algorithm != null ? algorithm.getName() : null);
        vo.setRating(rating.getRating());
        vo.setComment(rating.getComment());
        vo.setTags(parseList(rating.getTags()));
        vo.setImageUrls(parseList(rating.getImageUrls()));
        vo.setIsAnonymous(rating.getIsAnonymous());
        vo.setAdminReply(rating.getAdminReply());
        vo.setReplyTime(rating.getReplyTime());
        vo.setCreateTime(rating.getCreateTime());
        vo.setUserId(rating.getUserId());
        if (rating.getIsAnonymous() != null && rating.getIsAnonymous() == 1) {
            vo.setUserId(null);
            vo.setUsername(null);
            vo.setUserAvatar(null);
        } else {
            vo.setUsername(user != null ? user.getUsername() : null);
            vo.setUserAvatar(user != null ? user.getAvatar() : null);
        }
        vo.setIsHidden(rating.getIsHidden());
        vo.setAlgorithmId(rating.getAlgorithmId());
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

    private void invalidateRatingCaches(Long algorithmId) {
        try {
            stringRedisTemplate.delete(CACHE_RATING_STATS_GLOBAL);
            if (algorithmId != null) {
                stringRedisTemplate.delete("rating:stats:algorithm:" + algorithmId);
                stringRedisTemplate.delete("algorithm:rating:" + algorithmId);
            }
        } catch (Exception e) {
            log.warn("清除评价统计缓存失败", e);
        }
    }

    private void tryGrantGrowth(Long userId) {
        String key = "rating:daily:" + userId + ":" + LocalDate.now();
        String countStr = stringRedisTemplate.opsForValue().get(key);
        int count = countStr != null ? Integer.parseInt(countStr) : 0;
        if (count >= RATING_DAILY_GROWTH_LIMIT) {
            return;
        }
        memberService.adjustGrowth(userId, new MemberGrowthAdjustForm(RATING_GROWTH_VALUE, "评价奖励"));
        Long newCount = stringRedisTemplate.opsForValue().increment(key);
        if (newCount != null && newCount == 1L) {
            stringRedisTemplate.expire(key, Duration.ofHours(25));
        }
    }

    private void publishLowRatingAlert(Long ratingId) {
        RabbitMQPublisher publisher = rabbitMQPublisherProvider.getIfAvailable();
        if (publisher == null) {
            log.warn("RabbitMQ未启用，跳过低分告警: ratingId={}", ratingId);
            return;
        }
        try {
            String traceId = MDC.get(TraceIdFilter.MDC_TRACE_ID);
            publisher.publish("feedback.low_rating", String.valueOf(ratingId), traceId);
        } catch (Exception e) {
            log.warn("低分告警消息发送失败: ratingId={}", ratingId, e);
        }
    }
}
