package com.pei.dehaze.service.impl;

import cn.hutool.json.JSONObject;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysRecommendationMapper;
import com.pei.dehaze.mapper.SysRecommendationRuleMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysRecommendation;
import com.pei.dehaze.model.entity.SysRecommendationRule;
import com.pei.dehaze.model.form.AnalyzeForm;
import com.pei.dehaze.model.form.RecommendationFeedbackForm;
import com.pei.dehaze.model.form.RecommendationRuleForm;
import com.pei.dehaze.model.vo.IdVO;
import com.pei.dehaze.model.vo.ImageFeatureAnalysisVO;
import com.pei.dehaze.model.vo.RecommendationReportVO;
import com.pei.dehaze.model.vo.RecommendationRuleVO;
import com.pei.dehaze.model.vo.RecommendedAlgorithmVO;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.RecommendationService;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.client.PythonAlgorithmClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class RecommendationServiceImpl extends ServiceImpl<SysRecommendationMapper, SysRecommendation>
        implements RecommendationService {

    private static final List<String> VALID_SCENE_TYPES = List.of("urban", "landscape", "building", "night", "backlight", "indoor");
    private static final List<String> IMAGE_EXTENSIONS = List.of(".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif");
    private static final int TOP_N = 3;

    private final SysRecommendationRuleMapper ruleMapper;
    private final SysAlgorithmService sysAlgorithmService;
    private final PythonAlgorithmClient pythonAlgorithmClient;

    /**
     * 规则内存缓存，写时复制保证线程安全
     */
    private final CopyOnWriteArrayList<SysRecommendationRule> ruleCache = new CopyOnWriteArrayList<>();

    @Override
    public ImageFeatureAnalysisVO analyze(AnalyzeForm form) {
        String imageUrl = resolveImageUrl(form);
        validateImageFormat(imageUrl);

        // 调用 Python 图像特征分析服务提取真实特征
        // Python 服务不可用时由 PythonAlgorithmClient 抛出 BusinessException，不降级为伪特征，避免误导用户
        JSONObject data = pythonAlgorithmClient.analyzeImage(imageUrl);
        return toFeatureVO(data);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public List<RecommendedAlgorithmVO> getAlgorithmRecommendations(Long analysisId, String imageMd5) {
        // 获取已发布算法作为候选池
        List<SysAlgorithm> publishedAlgorithms = sysAlgorithmService.getAllAlgorithms().stream()
                .filter(a -> a.getStatus() != null && a.getStatus() == 4) // 4 = 已发布
                .toList();

        // 加载启用的规则
        List<SysRecommendationRule> rules = getEnabledRules();

        // 确定场景类型：优先从 analysisId 查历史记录，否则按 imageMd5 查，都没有则默认 urban
        String sceneType = "urban";
        if (analysisId != null && analysisId > 0) {
            SysRecommendation rec = this.getById(analysisId);
            if (rec != null && rec.getAnalysisResult() != null) {
                Object st = rec.getAnalysisResult().get("sceneType");
                if (st instanceof String s && VALID_SCENE_TYPES.contains(s)) {
                    sceneType = s;
                }
            }
        }
        if (imageMd5 != null && !imageMd5.isBlank() && "urban".equals(sceneType)) {
            SysRecommendation rec = this.getOne(new LambdaQueryWrapper<SysRecommendation>()
                    .eq(SysRecommendation::getImageMd5, imageMd5)
                    .orderByDesc(SysRecommendation::getId)
                    .last("LIMIT 1"));
            if (rec != null && rec.getAnalysisResult() != null) {
                Object st = rec.getAnalysisResult().get("sceneType");
                if (st instanceof String s && VALID_SCENE_TYPES.contains(s)) {
                    sceneType = s;
                }
            }
        }

        List<RecommendedAlgorithmVO> result = Collections.emptyList();

        // 仅在已发布算法和规则都就绪时才做匹配
        if (!publishedAlgorithms.isEmpty() && !rules.isEmpty()) {
            final String matchedScene = sceneType;
            List<SysRecommendationRule> matchedRules = rules.stream()
                    .filter(r -> r.getSceneType() != null && r.getSceneType().equals(matchedScene))
                    .sorted(Comparator.comparingInt(SysRecommendationRule::getWeight).reversed())
                    .toList();

            if (!matchedRules.isEmpty()) {
                Set<Long> candidateIds = matchedRules.stream()
                        .flatMap(r -> r.getAlgorithmIds() != null ? r.getAlgorithmIds().stream() : java.util.stream.Stream.empty())
                        .collect(Collectors.toSet());

                List<SysAlgorithm> candidates = publishedAlgorithms.stream()
                        .filter(a -> candidateIds.contains(a.getId()))
                        .toList();

                if (!candidates.isEmpty()) {
                    result = candidates.stream()
                            .map(alg -> {
                                RecommendedAlgorithmVO vo = new RecommendedAlgorithmVO();
                                vo.setAlgorithmId(alg.getId());
                                vo.setAlgorithmName(alg.getName());
                                int matchScore = computeMatchScore(alg.getId(), matchedRules);
                                vo.setMatchScore(matchScore);
                                vo.setReason(buildReason(matchedScene, alg.getName()));
                                vo.setEffectDescription("该算法在" + matchedScene + "场景下表现稳定");
                                return vo;
                            })
                            .sorted(Comparator.comparingInt(RecommendedAlgorithmVO::getMatchScore).reversed())
                            .limit(TOP_N)
                            .toList();
                }
            }
        }

        // 无论有无结果，都写入 sys_recommendation 记录，确保 feedback 能找到记录
        List<Map<String, Object>> topAlgorithms = result.stream()
                .map(vo -> {
                    Map<String, Object> item = new java.util.HashMap<>();
                    item.put("algorithmId", vo.getAlgorithmId());
                    item.put("algorithmName", vo.getAlgorithmName());
                    item.put("matchScore", vo.getMatchScore());
                    return item;
                })
                .toList();

        SysRecommendation rec = new SysRecommendation();
        rec.setUserId(currentUserId());
        rec.setImageMd5(imageMd5);
        rec.setTargetType("algorithm");
        rec.setTopAlgorithms(topAlgorithms);
        rec.setFeedback(0);
        this.save(rec);

        // 回填 recommendationId 到 VO
        final Long recommendationId = rec.getId();
        result.forEach(vo -> vo.setRecommendationId(recommendationId));

        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public IdVO submitFeedback(RecommendationFeedbackForm form) {
        SysRecommendation rec = this.getById(form.getRecommendationId());
        if (rec == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        rec.setFeedback(form.getUseful() ? 1 : 2);
        this.updateById(rec);
        return new IdVO(rec.getId());
    }

    @Override
    public List<RecommendationRuleVO> getRules() {
        List<SysRecommendationRule> rules = ruleMapper.selectList(
                new LambdaQueryWrapper<SysRecommendationRule>()
                        .orderByAsc(SysRecommendationRule::getWeight));
        return rules.stream().map(this::toRuleVO).toList();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Long updateRule(Long id, RecommendationRuleForm form) {
        // 权重校验
        if (form.getWeight() == null || form.getWeight() < 0 || form.getWeight() > 100) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "规则权重必须在0-100之间");
        }

        if (id == null || id == 0) {
            // 新增
            SysRecommendationRule rule = new SysRecommendationRule();
            rule.setRuleName(form.getRuleName());
            rule.setSceneType(form.getSceneType());
            rule.setAlgorithmIds(form.getAlgorithmIds());
            rule.setWeight(form.getWeight());
            rule.setEnabled(form.getEnabled() != null && form.getEnabled() ? 1 : 0);
            ruleMapper.insert(rule);
            refreshRuleCache();
            return rule.getId() != null ? rule.getId().longValue() : 0L;
        }

        // 更新
        SysRecommendationRule rule = ruleMapper.selectById(id);
        if (rule == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        rule.setRuleName(form.getRuleName());
        rule.setSceneType(form.getSceneType());
        rule.setAlgorithmIds(form.getAlgorithmIds());
        rule.setWeight(form.getWeight());
        rule.setEnabled(form.getEnabled() != null && form.getEnabled() ? 1 : 0);
        ruleMapper.updateById(rule);
        refreshRuleCache();
        return rule.getId() != null ? rule.getId().longValue() : id;
    }

    @Override
    public RecommendationReportVO getReport(String startDate, String endDate) {
        LocalDateTime start = parseDate(startDate, true);
        LocalDateTime end = parseDate(endDate, false);

        long total = this.baseMapper.countTotal(start, end);
        long usefulCount = this.baseMapper.countUseful(start, end);
        long feedbackTotal = this.baseMapper.countFeedbackTotal(start, end);
        long adoptedDistinct = this.baseMapper.countAdoptedAlgorithmDistinct(start, end);

        // 获取已发布算法总数
        List<SysAlgorithm> published = sysAlgorithmService.getAllAlgorithms().stream()
                .filter(a -> a.getStatus() != null && a.getStatus() == 4)
                .toList();
        long publishedCount = published.size();

        RecommendationReportVO vo = new RecommendationReportVO();
        vo.setTotalRecommendations(total);
        vo.setAdoptionRate(feedbackTotal > 0 ? (double) usefulCount / feedbackTotal : 0.0);
        // 满意度：本阶段简化，有用即满意
        vo.setSatisfactionRate(feedbackTotal > 0 ? (double) usefulCount / feedbackTotal : 0.0);
        vo.setCoverageRate(publishedCount > 0 ? (double) adoptedDistinct / publishedCount : 0.0);
        // 冷启动成功率：简化计算
        vo.setColdStartSuccessRate(0.0);

        // 趋势按日聚合
        List<Map<String, Object>> dailyData = this.baseMapper.selectDailyAdoptionRate(start, end);
        List<RecommendationReportVO.TrendItem> trend = new ArrayList<>();
        for (Map<String, Object> row : dailyData) {
            RecommendationReportVO.TrendItem item = new RecommendationReportVO.TrendItem();
            Object dateObj = row.get("date");
            item.setDate(dateObj != null ? dateObj.toString() : "");
            Object rateObj = row.get("adoptionRate");
            item.setAdoptionRate(rateObj != null ? ((Number) rateObj).doubleValue() : 0.0);
            trend.add(item);
        }
        vo.setTrend(trend);

        return vo;
    }

    // ==================== 内部方法 ====================

    private Long currentUserId() {
        Long userId = SecurityUtils.getUserId();
        return userId != null ? userId : SystemConstants.SYSTEM_USER_ID;
    }

    private String resolveImageUrl(AnalyzeForm form) {
        if (form.getImageId() != null && form.getImageId() > 0) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "imageId方式暂不支持，请使用imageUrl");
        }
        if (form.getImageUrl() != null && !form.getImageUrl().isBlank()) {
            return form.getImageUrl();
        }
        throw new BusinessException(ResultCode.PARAM_ERROR, "imageId和imageUrl至少提供一个");
    }

    private void validateImageFormat(String imageUrl) {
        String lower = imageUrl.toLowerCase();
        // 去掉 query string
        int qIdx = lower.indexOf('?');
        if (qIdx > 0) {
            lower = lower.substring(0, qIdx);
        }
        boolean valid = IMAGE_EXTENSIONS.stream().anyMatch(lower::endsWith);
        if (!valid) {
            throw new BusinessException(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH);
        }
    }

    /**
     * 将 Python 图像特征分析响应映射为 VO
     */
    private ImageFeatureAnalysisVO toFeatureVO(JSONObject data) {
        ImageFeatureAnalysisVO vo = new ImageFeatureAnalysisVO();
        vo.setImageMd5(data.getStr("imageMd5"));
        vo.setHazeLevel(data.getStr("hazeLevel"));
        vo.setHazeConfidence(data.getDouble("hazeConfidence"));
        vo.setSceneType(data.getStr("sceneType"));
        vo.setSceneConfidence(data.getDouble("sceneConfidence"));
        vo.setLighting(data.getStr("lighting"));
        vo.setComplexity(data.getDouble("complexity"));
        JSONObject cd = data.getJSONObject("colorDistribution");
        if (cd != null) {
            ImageFeatureAnalysisVO.ColorDistribution colorDistribution = new ImageFeatureAnalysisVO.ColorDistribution();
            colorDistribution.setTemperature(cd.getDouble("temperature"));
            colorDistribution.setSaturation(cd.getDouble("saturation"));
            vo.setColorDistribution(colorDistribution);
        }
        vo.setResolution(data.getStr("resolution"));
        vo.setNoiseLevel(data.getStr("noiseLevel"));
        return vo;
    }

    private List<SysRecommendationRule> getEnabledRules() {
        if (!ruleCache.isEmpty()) {
            return ruleCache;
        }
        return refreshRuleCache();
    }

    private List<SysRecommendationRule> refreshRuleCache() {
        List<SysRecommendationRule> rules = ruleMapper.selectList(
                new LambdaQueryWrapper<SysRecommendationRule>()
                        .eq(SysRecommendationRule::getEnabled, 1));
        ruleCache.clear();
        ruleCache.addAll(rules);
        return rules;
    }

    private int computeMatchScore(Long algorithmId, List<SysRecommendationRule> matchedRules) {
        // 取该算法在匹配规则中的最高权重作为匹配度
        int maxWeight = matchedRules.stream()
                .filter(r -> r.getAlgorithmIds() != null && r.getAlgorithmIds().contains(algorithmId))
                .mapToInt(SysRecommendationRule::getWeight)
                .max()
                .orElse(0);
        // 归一化到 0-100
        return Math.min(100, maxWeight);
    }

    private String buildReason(String sceneType, String algorithmName) {
        Map<String, String> reasonTemplates = Map.of(
                "urban", "处理速度快，对城市雾霾效果出色",
                "landscape", "在自然场景下表现稳定，色彩还原度高",
                "building", "深度模型，对建筑场景处理能力强",
                "night", "低光照增强组合，避免过度暗化",
                "backlight", "HDR预处理提升暗部细节",
                "indoor", "室内场景适配，细节保留好"
        );
        String reason = reasonTemplates.getOrDefault(sceneType, "综合表现优秀");
        return algorithmName + "：" + reason;
    }

    private RecommendationRuleVO toRuleVO(SysRecommendationRule entity) {
        RecommendationRuleVO vo = new RecommendationRuleVO();
        vo.setId(entity.getId());
        vo.setRuleName(entity.getRuleName());
        vo.setSceneType(entity.getSceneType());
        vo.setAlgorithmIds(entity.getAlgorithmIds());
        vo.setWeight(entity.getWeight());
        vo.setEnabled(entity.getEnabled() != null && entity.getEnabled() == 1);
        return vo;
    }

    private LocalDateTime parseDate(String dateStr, boolean startOfDay) {
        if (dateStr == null || dateStr.isBlank()) {
            return null;
        }
        try {
            LocalDate date = LocalDate.parse(dateStr, DateTimeFormatter.ISO_LOCAL_DATE);
            return startOfDay ? date.atStartOfDay() : date.atTime(23, 59, 59);
        } catch (Exception e) {
            return null;
        }
    }
}
