package com.pei.dehaze.service;

import com.pei.dehaze.model.form.AlgorithmCompareForm;
import com.pei.dehaze.model.form.AlgorithmTestForm;
import com.pei.dehaze.model.vo.AlgorithmCompareVO;
import com.pei.dehaze.model.vo.AlgorithmDetailVO;
import com.pei.dehaze.model.vo.AlgorithmSelectNodeVO;
import com.pei.dehaze.model.vo.PredictionResultVO;

import java.util.List;

public interface AlgorithmSelectService {

    /**
     * 获取算法选择树（仅已发布算法）
     */
    List<AlgorithmSelectNodeVO> getTree();

    /**
     * 获取算法详情（含样例效果图、评分、使用次数）
     */
    AlgorithmDetailVO getDetail(Long id);

    /**
     * 上传自定义图片测试算法效果
     */
    PredictionResultVO test(Long algorithmId, AlgorithmTestForm form);

    /**
     * 搜索算法（关键词/拼音/标签）
     */
    List<AlgorithmSelectNodeVO> search(String keyword);

    /**
     * 算法对比（最多3个）
     */
    List<AlgorithmCompareVO> compare(AlgorithmCompareForm form);
}
