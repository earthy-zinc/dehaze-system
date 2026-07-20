package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.converter.AlgorithmVersionConverter;
import com.pei.dehaze.mapper.SysAlgorithmVersionMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysAlgorithmVersion;
import com.pei.dehaze.model.form.AlgorithmVersionForm;
import com.pei.dehaze.model.vo.AlgorithmVersionVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysAlgorithmVersionService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * @author earthyzinc
 * @since 2024-06-12
 */
@Service
@RequiredArgsConstructor
public class SysAlgorithmVersionServiceImpl extends ServiceImpl<SysAlgorithmVersionMapper, SysAlgorithmVersion>
        implements SysAlgorithmVersionService {

    private final SysAlgorithmService algorithmService;
    private final AlgorithmVersionConverter algorithmVersionConverter;

    @Override
    public List<AlgorithmVersionVO> getVersionHistory(Long algorithmId) {
        List<SysAlgorithmVersion> versions = this.list(new LambdaQueryWrapper<SysAlgorithmVersion>()
                .eq(SysAlgorithmVersion::getAlgorithmId, algorithmId)
                .orderByDesc(SysAlgorithmVersion::getCreateTime));
        return versions.stream()
                .map(algorithmVersionConverter::entity2Vo)
                .toList();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public SysAlgorithmVersion addVersion(Long algorithmId, AlgorithmVersionForm form) {
        SysAlgorithm algorithm = algorithmService.getById(algorithmId);
        if (algorithm == null) {
            throw new BusinessException("算法不存在");
        }

        // 检查版本号是否已存在
        long count = this.count(new LambdaQueryWrapper<SysAlgorithmVersion>()
                .eq(SysAlgorithmVersion::getAlgorithmId, algorithmId)
                .eq(SysAlgorithmVersion::getVersion, form.getVersion()));
        if (count > 0) {
            throw new BusinessException("版本号 " + form.getVersion() + " 已存在");
        }

        // 将当前活跃版本置为非活跃
        deactivateCurrentActiveVersion(algorithmId);

        // 创建新版本记录
        SysAlgorithmVersion version = new SysAlgorithmVersion();
        version.setAlgorithmId(algorithmId);
        version.setVersion(form.getVersion());
        version.setChangeLog(form.getChangeLog());
        version.setStatus(algorithm.getStatus());
        version.setModelFileId(form.getModelFileId());
        version.setIsActive(true);
        this.save(version);

        // 更新主表版本号
        algorithm.setVersion(form.getVersion());
        algorithmService.updateById(algorithm);

        return version;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void rollbackToVersion(Long algorithmId, Long versionId) {
        SysAlgorithm algorithm = algorithmService.getById(algorithmId);
        if (algorithm == null) {
            throw new BusinessException("算法不存在");
        }

        SysAlgorithmVersion targetVersion = this.getById(versionId);
        if (targetVersion == null || !targetVersion.getAlgorithmId().equals(algorithmId)) {
            throw new BusinessException("版本不存在或不属于该算法");
        }

        // 如果已经是当前活跃版本，不允许回滚
        if (Boolean.TRUE.equals(targetVersion.getIsActive())) {
            throw new BusinessException("当前已是该版本，无需回滚");
        }

        // 将当前活跃版本置为非活跃
        deactivateCurrentActiveVersion(algorithmId);

        // 激活目标版本
        targetVersion.setIsActive(true);
        this.updateById(targetVersion);

        // 更新主表版本号
        algorithm.setVersion(targetVersion.getVersion());
        algorithmService.updateById(algorithm);
    }

    /**
     * 将指定算法的当前活跃版本置为非活跃
     */
    private void deactivateCurrentActiveVersion(Long algorithmId) {
        Long currentUserId = SecurityUtils.getUserId();
        this.update(new LambdaUpdateWrapper<SysAlgorithmVersion>()
                .eq(SysAlgorithmVersion::getAlgorithmId, algorithmId)
                .eq(SysAlgorithmVersion::getIsActive, true)
                .set(SysAlgorithmVersion::getIsActive, false)
                .set(SysAlgorithmVersion::getUpdateBy, currentUserId));
    }
}
