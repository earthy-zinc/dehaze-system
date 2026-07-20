package com.pei.dehaze.service.impl;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONObject;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.query.PredLogQuery;
import com.pei.dehaze.model.vo.PredLogVO;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysPredLogService;
import com.pei.dehaze.service.client.PythonAlgorithmClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 模型预测服务 —— 生产级实现
 * <p>
 * 调用 Python 算法服务进行实际去雾处理，带重试 + 熔断 + 日志记录
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysPredLogServiceImpl extends ServiceImpl<SysPredLogMapper, SysPredLog> implements SysPredLogService {

    private final SysAlgorithmService algorithmService;
    private final PythonAlgorithmClient pythonClient;
    private final SysFileService sysFileService;

    @Override
    public PredictionResultVO predict(PredictionForm form) {
        // 1. 校验算法存在且可用
        SysAlgorithm algorithm = algorithmService.getById(form.getAlgorithmId());
        if (algorithm == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
        }

        // 2. 确定图片来源 URL
        String imageUrl = resolveImageUrl(form);
        if (CharSequenceUtil.isBlank(imageUrl)) {
            throw new BusinessException("图片来源不能为空，请提供 fileId 或 imageUrl");
        }

        // 3. 记录预测请求日志（独立短事务，避免远程调用期间占用数据库连接）
        SysPredLog predLog = new SysPredLog();
        predLog.setAlgorithmId(form.getAlgorithmId());
        if (form.getFileId() != null) {
            predLog.setOriginFileId(form.getFileId());
        }
        predLog.setOriginUrl(imageUrl);
        this.save(predLog);

        // 4. 调用 Python 算法服务（事务外远程调用，不占用数据库连接）
        long startTime = System.currentTimeMillis();
        try {
            JSONObject result = pythonClient.predict(
                    form.getAlgorithmId(),
                    imageUrl,
                    form.getParams());

            // 5. 更新日志（成功，独立短事务）
            int elapsed = (int) (System.currentTimeMillis() - startTime);
            predLog.setTime(elapsed);
            predLog.setPredUrl(result.getStr("resultUrl"));
            predLog.setPredMd5(result.getStr("resultMd5"));
            this.updateById(predLog);

            // 6. 构造返回
            PredictionResultVO vo = new PredictionResultVO();
            vo.setLogId(predLog.getId());
            vo.setResultUrl(result.getStr("resultUrl"));
            vo.setResultThumbnailUrl(result.getStr("resultThumbnailUrl"));
            vo.setTime(elapsed);

            log.info("预测完成: algorithmId={}, predLogId={}, time={}ms",
                    form.getAlgorithmId(), predLog.getId(), elapsed);
            return vo;

        } catch (BusinessException e) {
            // 业务异常 —— 不重试，记录失败
            int elapsed = (int) (System.currentTimeMillis() - startTime);
            predLog.setTime(elapsed);
            this.updateById(predLog);
            log.error("预测失败: algorithmId={}, predLogId={}, error={}",
                    form.getAlgorithmId(), predLog.getId(), e.getMessage());
            throw e;
        }
    }

    @Override
    public Page<PredLogVO> getPredLogPage(PredLogQuery query) {
        Page<SysPredLog> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysPredLog> wrapper = new LambdaQueryWrapper<SysPredLog>()
                .eq(query.getAlgorithmId() != null, SysPredLog::getAlgorithmId, query.getAlgorithmId())
                .orderByDesc(SysPredLog::getCreateTime);

        Page<SysPredLog> result = this.page(page, wrapper);
        Page<PredLogVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());

        // 批量预加载算法名称，避免分页内逐条 getById 触发 N+1 查询
        Set<Long> algorithmIds = result.getRecords().stream()
                .map(SysPredLog::getAlgorithmId)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
        Map<Long, String> algorithmNameMap = algorithmIds.isEmpty()
                ? Collections.emptyMap()
                : algorithmService.listByIds(algorithmIds).stream()
                        .collect(Collectors.toMap(SysAlgorithm::getId, SysAlgorithm::getName, (a, b) -> a));

        voPage.setRecords(result.getRecords().stream().map(log -> {
            PredLogVO vo = new PredLogVO();
            BeanUtil.copyProperties(log, vo);
            vo.setAlgorithmName(algorithmNameMap.getOrDefault(log.getAlgorithmId(), "未知算法"));
            return vo;
        }).toList());
        return voPage;
    }

    /**
     * 解析图片来源 URL，优先使用 fileId 对应的文件 URL
     */
    private String resolveImageUrl(PredictionForm form) {
        if (form.getFileId() != null) {
            // 从文件管理模块获取文件的绝对 URL（Python 服务需要可直接访问的地址）
            com.pei.dehaze.model.entity.SysFile sysFile = sysFileService.getById(form.getFileId());
            if (sysFile != null && CharSequenceUtil.isNotBlank(sysFile.getUrl())) {
                return sysFile.getUrl();
            }
            log.warn("文件不存在或 URL 为空: fileId={}", form.getFileId());
            return null;
        }
        return form.getImageUrl();
    }
}
