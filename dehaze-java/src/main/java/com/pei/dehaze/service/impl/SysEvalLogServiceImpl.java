package com.pei.dehaze.service.impl;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.json.JSONObject;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.form.EvaluationForm;
import com.pei.dehaze.model.query.EvalLogQuery;
import com.pei.dehaze.model.vo.EvalLogVO;
import com.pei.dehaze.model.vo.EvaluationResultVO;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysEvalLogService;
import com.pei.dehaze.service.client.PythonAlgorithmClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 效果评估服务 —— 生产级实现
 * <p>
 * 调用 Python 算法服务执行 PSNR/SSIM/LPIPS/NIQE 等多维指标评估
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysEvalLogServiceImpl extends ServiceImpl<SysEvalLogMapper, SysEvalLog> implements SysEvalLogService {

    private final SysAlgorithmService algorithmService;
    private final PythonAlgorithmClient pythonClient;

    @Override
    public EvaluationResultVO evaluate(EvaluationForm form) {
        // 1. 校验算法存在
        SysAlgorithm algorithm = algorithmService.getById(form.getAlgorithmId());
        if (algorithm == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND.getMsg() + ": 算法不存在");
        }

        // 2. 记录评估请求日志（独立短事务，避免远程调用期间占用数据库连接）
        SysEvalLog evalLog = new SysEvalLog();
        evalLog.setAlgorithmId(form.getAlgorithmId());
        if (form.getPredFileId() != null) {
            evalLog.setPredFileId(form.getPredFileId());
        }
        if (form.getGtFileId() != null) {
            evalLog.setGtFileId(form.getGtFileId());
        }
        this.save(evalLog);

        // 3. 调用 Python 评估服务（事务外远程调用，不占用数据库连接）
        long startTime = System.currentTimeMillis();
        try {
            String predUrl = resolveFileUrl(form.getPredFileId(), "pred");
            String gtUrl = resolveFileUrl(form.getGtFileId(), "gt");

            JSONObject result = pythonClient.evaluate(
                    form.getAlgorithmId(), predUrl, gtUrl);

            // 4. 更新日志（成功）
            int elapsed = (int) (System.currentTimeMillis() - startTime);
            evalLog.setTime(elapsed);
            evalLog.setResult(result.toString());
            this.updateById(evalLog);

            // 5. 解析评估指标
            Map<String, Double> metrics = new LinkedHashMap<>();
            JSONObject metricsJson = result.getJSONObject("metrics");
            if (metricsJson != null) {
                for (String key : metricsJson.keySet()) {
                    metrics.put(key, metricsJson.getDouble(key));
                }
            }

            EvaluationResultVO vo = new EvaluationResultVO();
            vo.setLogId(evalLog.getId());
            vo.setMetrics(metrics);
            vo.setTime(elapsed);

            log.info("评估完成: algorithmId={}, evalLogId={}, time={}ms, metrics={}",
                    form.getAlgorithmId(), evalLog.getId(), elapsed, metrics);
            return vo;

        } catch (BusinessException e) {
            int elapsed = (int) (System.currentTimeMillis() - startTime);
            evalLog.setTime(elapsed);
            this.updateById(evalLog);
            log.error("评估失败: algorithmId={}, evalLogId={}, error={}",
                    form.getAlgorithmId(), evalLog.getId(), e.getMessage());
            throw e;
        }
    }

    @Override
    public Page<EvalLogVO> getEvalLogPage(EvalLogQuery query) {
        Page<SysEvalLog> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysEvalLog> wrapper = new LambdaQueryWrapper<SysEvalLog>()
                .eq(query.getAlgorithmId() != null, SysEvalLog::getAlgorithmId, query.getAlgorithmId())
                .orderByDesc(SysEvalLog::getCreateTime);

        Page<SysEvalLog> result = this.page(page, wrapper);
        Page<EvalLogVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());
        voPage.setRecords(result.getRecords().stream().map(log -> {
            EvalLogVO vo = new EvalLogVO();
            BeanUtil.copyProperties(log, vo);
            SysAlgorithm algorithm = algorithmService.getById(log.getAlgorithmId());
            vo.setAlgorithmName(algorithm != null ? algorithm.getName() : "未知算法");
            return vo;
        }).toList());
        return voPage;
    }

    private String resolveFileUrl(Long fileId, String type) {
        if (fileId != null) {
            return "/api/v1/files/download/" + fileId;
        }
        throw new BusinessException("缺少" + ("pred".equals(type) ? "预测" : "参考") + "图片");
    }
}
