package com.pei.dehaze.service.impl;

import cn.hutool.core.bean.BeanUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysInputHistoryMapper;
import com.pei.dehaze.model.entity.SysInputHistory;
import com.pei.dehaze.model.form.HistoryForm;
import com.pei.dehaze.model.form.HistoryUpdateForm;
import com.pei.dehaze.model.query.HistoryQuery;
import com.pei.dehaze.model.vo.InputHistoryVO;
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

    /** 注册用户默认配额 */
    private static final int DEFAULT_QUOTA = 100;

    @Override
    public Page<InputHistoryVO> getHistoryPage(HistoryQuery query) {
        Long userId = getCurrentUserId();

        Page<SysInputHistory> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysInputHistory> wrapper = new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId)
                .eq(query.getStatus() != null, SysInputHistory::getStatus, query.getStatus())
                .eq(query.getInputSource() != null, SysInputHistory::getInputSource, query.getInputSource())
                .eq(query.getIsFavorite() != null && query.getIsFavorite(), SysInputHistory::getIsFavorite, true)
                .orderByDesc(SysInputHistory::getIsFavorite)
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
        if (history == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "历史记录不存在");
        }
        // 校验归属
        checkOwnership(history);
        InputHistoryVO vo = new InputHistoryVO();
        BeanUtil.copyProperties(history, vo);
        return vo;
    }

    @Override
    public Long createHistory(HistoryForm form) {
        Long userId = getCurrentUserId();

        // 配额检查
        long count = this.count(new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId));
        if (count >= DEFAULT_QUOTA) {
            // 自动清理最旧的非收藏记录
            autoCleanup(userId);
        }

        SysInputHistory history = new SysInputHistory();
        BeanUtil.copyProperties(form, history);
        history.setUserId(userId);
        history.setIsFavorite(false);
        history.setSyncStatus(0);
        if (form.getStatus() == null) {
            history.setStatus(3); // 默认处理中
        }
        this.save(history);
        return history.getId();
    }

    @Override
    public boolean updateHistory(Long id, HistoryUpdateForm form) {
        SysInputHistory history = this.getById(id);
        if (history == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "历史记录不存在");
        }
        checkOwnership(history);

        if (form.getIsFavorite() != null) {
            history.setIsFavorite(form.getIsFavorite());
        }
        return this.updateById(history);
    }

    @Override
    public boolean deleteHistory(Long id) {
        SysInputHistory history = this.getById(id);
        if (history == null) {
            return true; // 幂等
        }
        checkOwnership(history);
        return this.removeById(id);
    }

    @Override
    public int batchDeleteHistory(List<Long> ids) {
        Long userId = getCurrentUserId();
        if (ids == null || ids.isEmpty()) {
            return 0;
        }
        LambdaUpdateWrapper<SysInputHistory> wrapper = new LambdaUpdateWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId)
                .in(SysInputHistory::getId, ids);
        boolean result = this.remove(wrapper);
        return result ? ids.size() : 0;
    }

    @Override
    public int clearAllHistory() {
        Long userId = getCurrentUserId();
        LambdaQueryWrapper<SysInputHistory> wrapper = new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId);
        long count = this.count(wrapper);
        this.remove(wrapper);
        log.info("用户 {} 清空了 {} 条历史记录", userId, count);
        return (int) count;
    }

    @Override
    public int syncHistory() {
        // 当前版本：标记所有未同步记录为已同步
        Long userId = getCurrentUserId();
        LambdaUpdateWrapper<SysInputHistory> wrapper = new LambdaUpdateWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId)
                .eq(SysInputHistory::getSyncStatus, 0)
                .set(SysInputHistory::getSyncStatus, 1);
        boolean result = this.update(wrapper);
        log.info("用户 {} 同步历史记录完成", userId);
        return result ? 1 : 0;
    }

    // ==================== 内部方法 ====================

    private void checkOwnership(SysInputHistory history) {
        Long userId = getCurrentUserId();
        if (!history.getUserId().equals(userId)) {
            throw new BusinessException("无权限访问此记录");
        }
    }

    private void autoCleanup(Long userId) {
        // 删除最旧的非收藏记录
        LambdaQueryWrapper<SysInputHistory> wrapper = new LambdaQueryWrapper<SysInputHistory>()
                .eq(SysInputHistory::getUserId, userId)
                .eq(SysInputHistory::getIsFavorite, false)
                .orderByAsc(SysInputHistory::getCreateTime)
                .last("LIMIT 1");
        SysInputHistory oldest = this.getOne(wrapper);
        if (oldest != null) {
            this.removeById(oldest.getId());
            log.info("配额已满，自动清理最旧记录: id={}", oldest.getId());
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
