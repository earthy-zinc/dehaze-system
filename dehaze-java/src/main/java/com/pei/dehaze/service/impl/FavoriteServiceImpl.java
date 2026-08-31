package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.mapper.SysFavoriteMapper;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysFavorite;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.form.FavoriteForm;
import com.pei.dehaze.model.query.FavoritePageQuery;
import com.pei.dehaze.model.vo.FavoriteCountVO;
import com.pei.dehaze.model.vo.FavoriteStatusVO;
import com.pei.dehaze.model.vo.FavoriteVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.FavoriteService;
import com.pei.dehaze.service.SysDictService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class FavoriteServiceImpl extends ServiceImpl<SysFavoriteMapper, SysFavorite> implements FavoriteService {

    /** 收藏类型常量 */
    private static final List<String> VALID_TARGET_TYPES = Arrays.asList("algorithm", "result", "dataset", "image", "preset");

    /** 收藏容量字典类型 */
    private static final String FAVORITE_CAPACITY_DICT_TYPE = "favorite_capacity";

    private final SysMemberMapper memberMapper;
    private final SysAlgorithmMapper sysAlgorithmMapper;
    private final SysDatasetMapper sysDatasetMapper;
    private final SysPredLogMapper sysPredLogMapper;
    private final SysDictService sysDictService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Long add(FavoriteForm form) {
        Long userId = SecurityUtils.getUserId();

        // 校验收藏目标对象是否存在（仅校验已实现的类型，image/preset 预留类型跳过）
        validateTargetExists(form.getTargetType(), form.getTargetId());

        // 容量校验（MP 自动过滤 deleted=0）
        long currentCount = this.count(new LambdaQueryWrapper<SysFavorite>()
                .eq(SysFavorite::getUserId, userId));
        int capacity = getCapacity(userId);
        if (currentCount >= capacity) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "收藏已达上限（" + capacity + "条），请清理后重试");
        }

        SysFavorite favorite = new SysFavorite();
        favorite.setUserId(userId);
        favorite.setTargetType(form.getTargetType());
        favorite.setTargetId(form.getTargetId());
        favorite.setIsInvalid(0);
        this.baseMapper.upsertByUserAndTarget(favorite);
        return favorite.getId();
    }

    /**
     * 校验收藏目标对象是否存在
     * algorithm/dataset/result 为已实现类型，必须校验；image/preset 为预留类型，跳过校验
     */
    private void validateTargetExists(String targetType, Long targetId) {
        if (targetId == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "收藏目标不存在");
        }
        switch (targetType) {
            case "algorithm":
                if (sysAlgorithmMapper.selectById(targetId) == null) {
                    throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
                }
                break;
            case "dataset":
                if (sysDatasetMapper.selectById(targetId) == null) {
                    throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在");
                }
                break;
            case "result":
                if (sysPredLogMapper.selectById(targetId) == null) {
                    throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "处理记录不存在");
                }
                break;
            default:
                // image/preset 等预留类型，跳过校验
                break;
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteByIds(List<Long> ids) {
        if (ids == null || ids.isEmpty()) {
            return;
        }
        Long userId = SecurityUtils.getUserId();
        LambdaUpdateWrapper<SysFavorite> wrapper = new LambdaUpdateWrapper<>();
        wrapper.in(SysFavorite::getId, ids)
                .eq(SysFavorite::getUserId, userId)
                .set(SysFavorite::getDeleted, 1);
        this.update(wrapper);
    }

    @Override
    public Page<FavoriteVO> getPage(FavoritePageQuery query) {
        Long userId = SecurityUtils.getUserId();
        Page<FavoriteVO> page = new Page<>(query.getPageNum(), query.getPageSize());
        return this.baseMapper.selectFavoritePage(
                page,
                userId,
                query.getTargetType(),
                query.getKeywords(),
                query.getSortBy(),
                query.getSortOrder());
    }

    @Override
    public FavoriteStatusVO getStatus(String targetType, Long targetId) {
        Long userId = SecurityUtils.getUserId();
        long count = this.count(new LambdaQueryWrapper<SysFavorite>()
                .eq(SysFavorite::getUserId, userId)
                .eq(SysFavorite::getTargetType, targetType)
                .eq(SysFavorite::getTargetId, targetId));
        return new FavoriteStatusVO(targetType, targetId, count > 0);
    }

    @Override
    public List<FavoriteCountVO> getCount(String targetType) {
        Long userId = SecurityUtils.getUserId();
        LambdaQueryWrapper<SysFavorite> wrapper = new LambdaQueryWrapper<SysFavorite>()
                .eq(SysFavorite::getUserId, userId)
                .select(SysFavorite::getTargetType);
        if (CharSequenceUtil.isNotBlank(targetType)) {
            wrapper.eq(SysFavorite::getTargetType, targetType);
        }

        List<SysFavorite> list = this.list(wrapper);
        Map<String, Long> countMap = list.stream()
                .collect(Collectors.groupingBy(SysFavorite::getTargetType, Collectors.counting()));

        List<FavoriteCountVO> result = new ArrayList<>();
        for (String type : VALID_TARGET_TYPES) {
            if (targetType != null && !targetType.equals(type)) {
                continue;
            }
            result.add(new FavoriteCountVO(type, countMap.getOrDefault(type, 0L)));
        }
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void markInvalid(String targetType, List<Long> targetIds) {
        if (targetIds == null || targetIds.isEmpty()) {
            return;
        }
        LambdaUpdateWrapper<SysFavorite> wrapper = new LambdaUpdateWrapper<SysFavorite>()
                .eq(SysFavorite::getTargetType, targetType)
                .in(SysFavorite::getTargetId, targetIds)
                .set(SysFavorite::getIsInvalid, 1);
        this.update(wrapper);
    }

    /**
     * 根据用户会员等级返回收藏容量上限。
     * <p>容量由字典 {@code favorite_capacity} 维护，按等级映射字典键实时读取
     * （level_0→default、level_1→vip1、level_2→vip2、level_3→svip）。</p>
     */
    int getCapacity(Long userId) {
        SysMember member = memberMapper.selectOne(
                new LambdaQueryWrapper<SysMember>().eq(SysMember::getUserId, userId));
        String levelCode = member != null ? member.getLevelCode() : "level_0";
        String dictKey = switch (levelCode) {
            case "level_1" -> "vip1";
            case "level_2" -> "vip2";
            case "level_3" -> "svip";
            default -> "default";
        };
        return sysDictService.getIntValue(FAVORITE_CAPACITY_DICT_TYPE, dictKey, 200);
    }
}
