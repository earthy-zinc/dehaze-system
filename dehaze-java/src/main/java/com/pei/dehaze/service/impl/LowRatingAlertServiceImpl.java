package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.mapper.SysRatingMapper;
import com.pei.dehaze.mapper.SysRoleMapper;
import com.pei.dehaze.mapper.SysUserRoleMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysRating;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysUserRole;
import com.pei.dehaze.model.form.MessageSendForm;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.service.LowRatingAlertService;
import com.pei.dehaze.service.MessageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class LowRatingAlertServiceImpl implements LowRatingAlertService {

    private static final int LOW_RATING_THRESHOLD = 2;
    private static final int URGENT_LOW_RATING_COUNT = 3;
    private static final double SEVERE_LOW_RATING_RATE = 20.0;

    private final SysRatingMapper ratingMapper;
    private final SysAlgorithmMapper algorithmMapper;
    private final SysRoleMapper roleMapper;
    private final SysUserRoleMapper userRoleMapper;
    private final MessageService messageService;

    @Override
    public boolean checkAndAlert(SysRating rating) {
        if (rating == null || rating.getRating() == null || rating.getRating() > LOW_RATING_THRESHOLD) {
            return false;
        }
        sendNormalAlert(rating);
        if (rating.getRating() == 1) {
            if (countLowRatingsIn24Hours(rating.getAlgorithmId()) >= URGENT_LOW_RATING_COUNT) {
                sendUrgentAlert(rating.getAlgorithmId());
            }
        }
        if (calcDailyGlobalLowRatingRate() > SEVERE_LOW_RATING_RATE) {
            sendSevereAlert();
        }
        return true;
    }

    @Override
    public boolean sendNormalAlert(SysRating rating) {
        List<Long> adminUserIds = getAdminUserIds();
        if (adminUserIds.isEmpty()) {
            log.warn("低分告警：未找到管理员用户，跳过发送。ratingId={}", rating.getId());
            return false;
        }
        SysAlgorithm algorithm = rating.getAlgorithmId() != null
                ? algorithmMapper.selectById(rating.getAlgorithmId()) : null;
        String algorithmName = algorithm != null ? algorithm.getName() : String.valueOf(rating.getAlgorithmId());
        String title = "低分评价提醒";
        String content = String.format("用户对算法[%s]的评分较低（%d星），处理记录ID：%d，请关注。",
                algorithmName, rating.getRating(), rating.getPredLogId());

        MessageSendForm form = new MessageSendForm();
        form.setType("alert");
        form.setTitle(title);
        form.setContent(content);
        form.setRecipientIds(adminUserIds);
        form.setBizModule("rating_alert");
        form.setBizId(rating.getId() + ":normal");
        form.setPriority(2);
        messageService.send(form);
        log.info("低分普通告警已发送: ratingId={}, rating={}", rating.getId(), rating.getRating());
        return true;
    }

    @Override
    public boolean sendUrgentAlert(Long algorithmId) {
        List<Long> adminUserIds = getAdminUserIds();
        if (adminUserIds.isEmpty()) {
            log.warn("紧急告警：未找到管理员用户，跳过发送。algorithmId={}", algorithmId);
            return false;
        }
        SysAlgorithm algorithm = algorithmMapper.selectById(algorithmId);
        String algorithmName = algorithm != null ? algorithm.getName() : String.valueOf(algorithmId);
        String title = "算法低分紧急告警";
        String content = String.format("算法[%s]在24小时内收到%d条以上低分评价，请紧急关注。", algorithmName, URGENT_LOW_RATING_COUNT);

        MessageSendForm form = new MessageSendForm();
        form.setType("alert");
        form.setTitle(title);
        form.setContent(content);
        form.setRecipientIds(adminUserIds);
        form.setBizModule("rating_alert");
        form.setBizId(algorithmId + ":urgent");
        form.setPriority(1);
        messageService.send(form);
        log.info("低分紧急告警已发送: algorithmId={}", algorithmId);
        return true;
    }

    @Override
    public boolean sendSevereAlert() {
        List<Long> adminUserIds = getAdminUserIds();
        if (adminUserIds.isEmpty()) {
            log.warn("严重告警：未找到管理员用户，跳过发送。");
            return false;
        }
        String title = "全局低分率严重告警";
        String content = String.format("当日全局低分率超过%.0f%%，请立即关注算法质量。", SEVERE_LOW_RATING_RATE);

        MessageSendForm form = new MessageSendForm();
        form.setType("critical_alert");
        form.setTitle(title);
        form.setContent(content);
        form.setRecipientIds(adminUserIds);
        form.setBizModule("rating_alert");
        form.setBizId(LocalDate.now() + ":severe");
        form.setPriority(1);
        messageService.send(form);
        log.info("低分严重告警已发送");
        return true;
    }

    private long countLowRatingsIn24Hours(Long algorithmId) {
        if (algorithmId == null) {
            return 0;
        }
        return ratingMapper.selectCount(new LambdaQueryWrapper<SysRating>()
                .eq(SysRating::getAlgorithmId, algorithmId)
                .le(SysRating::getRating, LOW_RATING_THRESHOLD)
                .ge(SysRating::getCreateTime, LocalDateTime.now().minusHours(24)));
    }

    private double calcDailyGlobalLowRatingRate() {
        LocalDateTime dayStart = LocalDate.now().atStartOfDay();
        LocalDateTime dayEnd = LocalDate.now().atTime(23, 59, 59);
        List<SysRating> todayRatings = ratingMapper.selectList(new LambdaQueryWrapper<SysRating>()
                .ge(SysRating::getCreateTime, dayStart)
                .le(SysRating::getCreateTime, dayEnd));
        if (todayRatings.isEmpty()) {
            return 0.0;
        }
        long lowCount = todayRatings.stream()
                .filter(r -> r.getRating() != null && r.getRating() <= LOW_RATING_THRESHOLD)
                .count();
        return (double) lowCount * 100 / todayRatings.size();
    }

    private List<Long> getAdminUserIds() {
        SysRole rootRole = roleMapper.selectOne(new LambdaQueryWrapper<SysRole>()
                .eq(SysRole::getCode, SystemConstants.ROOT_ROLE_CODE));
        if (rootRole == null) {
            return Collections.emptyList();
        }
        List<SysUserRole> userRoles = userRoleMapper.selectList(new LambdaQueryWrapper<SysUserRole>()
                .eq(SysUserRole::getRoleId, rootRole.getId()));
        return userRoles.stream().map(SysUserRole::getUserId).distinct().toList();
    }
}
