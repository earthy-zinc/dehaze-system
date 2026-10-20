package com.pei.clarity.service;

import com.pei.clarity.model.form.AnalyzeForm;
import com.pei.clarity.model.form.RecommendationFeedbackForm;
import com.pei.clarity.model.form.RecommendationRuleForm;
import com.pei.clarity.model.vo.IdVO;
import com.pei.clarity.model.vo.ImageFeatureAnalysisVO;
import com.pei.clarity.model.vo.RecommendationReportVO;
import com.pei.clarity.model.vo.RecommendationRuleVO;
import com.pei.clarity.model.vo.RecommendedModelVO;
import java.util.List;

public interface RecommendationService {

    /**
     * 图像特征分析（F-REC-001）
     */
    ImageFeatureAnalysisVO analyze(AnalyzeForm form);

    /**
     * 获取图像模型推荐（F-REC-002）
     */
    List<RecommendedModelVO> getModelRecommendations(Long analysisId, String imageMd5);

    /**
     * 提交推荐反馈（F-REC-003）
     */
    IdVO submitFeedback(RecommendationFeedbackForm form);

    /**
     * 获取推荐规则列表（管理员）
     */
    List<RecommendationRuleVO> getRules();

    /**
     * 更新推荐规则（管理员，upsert）
     */
    Long updateRule(Long id, RecommendationRuleForm form);

    /**
     * 获取推荐效果报表（管理员）
     */
    RecommendationReportVO getReport(String startDate, String endDate);
}
