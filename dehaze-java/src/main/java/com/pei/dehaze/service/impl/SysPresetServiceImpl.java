package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysPresetMapper;
import com.pei.dehaze.model.entity.SysPreset;
import com.pei.dehaze.model.form.PresetForm;
import com.pei.dehaze.model.vo.PresetVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysPresetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class SysPresetServiceImpl extends ServiceImpl<SysPresetMapper, SysPreset> implements SysPresetService {

    private static final String TYPE_SYSTEM = "system";
    private static final String TYPE_CUSTOM = "custom";

    @Override
    public Page<PresetVO> listPresets(Long algorithmId, int pageNum, int pageSize, Boolean isSystem) {
        Long userId = SecurityUtils.getUserId();

        LambdaQueryWrapper<SysPreset> wrapper = new LambdaQueryWrapper<SysPreset>()
                .eq(algorithmId != null, SysPreset::getAlgorithmId, algorithmId);

        if (isSystem != null && isSystem) {
            // 仅系统预设
            wrapper.eq(SysPreset::getType, TYPE_SYSTEM);
        } else if (isSystem != null && !isSystem) {
            // 仅用户自定义预设
            wrapper.eq(SysPreset::getType, TYPE_CUSTOM)
                   .eq(SysPreset::getUserId, userId);
        } else {
            // 全部：系统预设 + 用户自定义
            wrapper.and(w -> w.eq(SysPreset::getType, TYPE_SYSTEM)
                    .or().eq(SysPreset::getType, TYPE_CUSTOM)
                    .eq(SysPreset::getUserId, userId));
        }

        wrapper.orderByAsc(SysPreset::getCreateTime);

        Page<SysPreset> page = this.page(new Page<>(pageNum, pageSize), wrapper);
        Page<PresetVO> voPage = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        voPage.setRecords(page.getRecords().stream().map(this::toVO).collect(Collectors.toList()));
        return voPage;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PresetVO createPreset(PresetForm form) {
        Long userId = SecurityUtils.getUserId();

        SysPreset preset = new SysPreset();
        preset.setName(form.getName());
        preset.setType(TYPE_CUSTOM);
        preset.setAlgorithmId(form.getAlgorithmId());
        preset.setParams(form.getParams());
        preset.setUserId(userId);
        preset.setIsDefault(form.getIsDefault() != null ? form.getIsDefault() : 0);
        this.save(preset);

        return toVO(preset);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PresetVO updatePreset(Long id, PresetForm form) {
        Long userId = SecurityUtils.getUserId();
        SysPreset preset = this.getById(id);
        if (preset == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "预设不存在");
        }
        if (TYPE_SYSTEM.equals(preset.getType())) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "系统预设不可修改");
        }
        if (!preset.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "不能操作他人的预设");
        }

        preset.setName(form.getName());
        preset.setAlgorithmId(form.getAlgorithmId());
        preset.setParams(form.getParams());
        preset.setIsDefault(form.getIsDefault() != null ? form.getIsDefault() : preset.getIsDefault());
        this.updateById(preset);

        return toVO(preset);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deletePreset(Long id) {
        Long userId = SecurityUtils.getUserId();
        SysPreset preset = this.getById(id);
        if (preset == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "预设不存在");
        }
        if (TYPE_SYSTEM.equals(preset.getType())) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "系统预设不可删除");
        }
        if (!preset.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "不能操作他人的预设");
        }

        this.removeById(id);
    }

    private PresetVO toVO(SysPreset entity) {
        PresetVO vo = new PresetVO();
        vo.setId(entity.getId());
        vo.setName(entity.getName());
        vo.setType(entity.getType());
        vo.setAlgorithmId(entity.getAlgorithmId());
        vo.setParams(entity.getParams());
        vo.setUserId(entity.getUserId());
        vo.setIsDefault(entity.getIsDefault());
        vo.setCreateTime(entity.getCreateTime());
        return vo;
    }
}
