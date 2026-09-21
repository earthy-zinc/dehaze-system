package com.pei.dehaze.service.impl;

import cn.hutool.core.bean.BeanUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysInputHistoryMapper;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.model.entity.SysInputHistory;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.form.HistoryForm;
import com.pei.dehaze.model.query.HistoryQuery;
import com.pei.dehaze.model.vo.InputHistoryVO;
import com.pei.dehaze.service.MemberBenefitService;
import com.pei.dehaze.service.SysInputHistoryService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 图像输入历史记录服务实现
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysInputHistoryServiceImpl extends ServiceImpl<SysInputHistoryMapper, SysInputHistory>
        implements SysInputHistoryService {

    /** 无会员档案/权益缺失时的兜底保留条数（对齐 sys_member_benefit level_0 种子值） */
    private static final int DEFAULT_QUOTA = 100;

    private final SysMemberMapper memberMapper;
    private final MemberBenefitService memberBenefitService;

    @Override
    public Page<InputHistoryVO> getHistoryPage(HistoryQuery query) {
        Long userId = getCurrentUserId();

        Page<SysInputHistory> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysInputHistory> wrapper = new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId)
                .eq(query.getStatus() != null, SysInputHistory::getStatus, query.getStatus())
                .eq(query.getInputSource() != null, SysInputHistory::getInputSource, query.getInputSource())
                .orderByDesc(SysInputHistory::getCreateTime);

        Page<SysInputHistory> result = this.page(page, wrapper);
        Page<InputHistoryVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());
        voPage.setRecords(result.getRecords().stream().map(h -> {
            InputHistoryVO vo = new InputHistoryVO();
            BeanUtil.copyProperties(h, vo);
            return vo;
        }).toList());
        return voPage;
    }

    @Override
    public InputHistoryVO getHistoryById(Long id) {
        SysInputHistory history = this.getById(id);
        // 非本人记录与不存在同样返回 404，不泄露记录存在性
        if (history == null || !history.getUserId().equals(getCurrentUserId())) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "历史记录不存在");
        }
        InputHistoryVO vo = new InputHistoryVO();
        BeanUtil.copyProperties(history, vo);
        return vo;
    }

    @Override
    public Long createHistory(HistoryForm form) {
        // algorithmParams 须为合法 JSON（python A0400 口径）
        if (cn.hutool.core.util.StrUtil.isNotBlank(form.getAlgorithmParams())) {
            try {
                cn.hutool.json.JSONUtil.parse(form.getAlgorithmParams());
            } catch (Exception e) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "algorithmParams 必须为合法 JSON 字符串");
            }
        }
        Long userId = getCurrentUserId();

        // 配额检查：超过会员等级保留条数时自动清理最旧记录
        long count = this.count(new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId));
        if (count >= getHistoryRetention(userId)) {
            autoCleanup(userId);
        }

        SysInputHistory history = new SysInputHistory();
        BeanUtil.copyProperties(form, history);
        history.setUserId(userId);
        if (form.getStatus() == null) {
            history.setStatus(3); // 默认处理中
        }
        this.save(history);
        return history.getId();
    }

    @Override
    public boolean deleteHistory(Long id) {
        SysInputHistory history = this.getById(id);
        // 幂等：不存在或非本人记录均静默成功（不泄露记录存在性）
        if (history == null || !history.getUserId().equals(getCurrentUserId())) {
            return true;
        }
        return this.removeById(id);
    }

    @Override
    public int batchDeleteHistory(List<Long> ids) {
        Long userId = getCurrentUserId();
        if (ids == null || ids.isEmpty()) {
            return 0;
        }
        LambdaQueryWrapper<SysInputHistory> wrapper = new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId)
                .in(SysInputHistory::getId, ids);
        // 返回实际删除数量（仅删除当前用户本人记录）
        return this.getBaseMapper().delete(wrapper);
    }

    @Override
    public int clearAllHistory() {
        Long userId = getCurrentUserId();
        LambdaQueryWrapper<SysInputHistory> wrapper = new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId);
        long count = this.count(wrapper);
        this.remove(wrapper);
        log.debug("用户 {} 清空了 {} 条历史记录", userId, count);
        return (int) count;
    }

    // ==================== 内部方法 ====================

    /**
     * 按会员等级取历史保留条数（sys_member_benefit.history_retention），
     * 无会员档案或权益缺失时回退默认值。
     */
    private int getHistoryRetention(Long userId) {
        SysMember member = memberMapper.selectOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId)
                .last("LIMIT 1"));
        if (member == null) {
            return DEFAULT_QUOTA;
        }
        var benefit = memberBenefitService.getByLevelCode(member.getLevelCode());
        if (benefit == null || benefit.getHistoryRetention() == null || benefit.getHistoryRetention() <= 0) {
            return DEFAULT_QUOTA;
        }
        return benefit.getHistoryRetention();
    }

    private void autoCleanup(Long userId) {
        // 删除最旧的记录
        LambdaQueryWrapper<SysInputHistory> wrapper = new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId)
                .orderByAsc(SysInputHistory::getCreateTime)
                .last("LIMIT 1");
        SysInputHistory oldest = this.getOne(wrapper);
        if (oldest != null) {
            this.removeById(oldest.getId());
            log.debug("配额已满，自动清理最旧记录: id={}", oldest.getId());
        }
    }

    private Long getCurrentUserId() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.getPrincipal() instanceof com.pei.dehaze.security.model.SysUserDetails userDetails) {
            return userDetails.getUserId();
        }
        throw new BusinessException("未获取到用户信息，请先登录");
    }
}
