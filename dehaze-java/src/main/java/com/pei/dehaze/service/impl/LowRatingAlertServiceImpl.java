package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysRatingMapper;
import com.pei.dehaze.mapper.SysRoleMapper;
import com.pei.dehaze.mapper.SysUserRoleMapper;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.entity.SysRating;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysUserRole;
import com.pei.dehaze.model.form.MessageSendForm;
import com.pei.dehaze.service.LowRatingAlertService;
import com.pei.dehaze.service.MessageService;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class LowRatingAlertServiceImpl implements LowRatingAlertService {

    private static final int LOW_RATING_THRESHOLD = 2;
    private static final int URGENT_LOW_RATING_COUNT = 3;
    private static final double SEVERE_LOW_RATING_RATE = 20.0;

    private final SysRatingMapper ratingMapper;
    private final SysAiModelMapper aiModelMapper;
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
            if (countLowRatingsIn24Hours(rating.getModelId()) >= URGENT_LOW_RATING_COUNT) {
                sendUrgentAlert(rating.getModelId());
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
        String modelName = resolveModelName(rating.getModelId());
        String title = "低分评价提醒";
        String content = String.format(
                "用户对模型[%s]的评分较低（%d星），处理记录ID：%d，请关注。", modelName, rating.getRating(), rating.getPredLogId());

        MessageSendForm form = new MessageSendForm();
        form.setType("alert");
        form.setTitle(title);
        form.setContent(content);
        form.setRecipientIds(adminUserIds);
        form.setBizModule("feedback");
        form.setBizId(rating.getId() + ":normal");
        form.setPriority(3);
        messageService.send(form);
        log.info("低分普通告警已发送: ratingId={}, rating={}", rating.getId(), rating.getRating());
        return true;
    }

    @Override
    public boolean sendUrgentAlert(String modelId) {
        List<Long> adminUserIds = getAdminUserIds();
        if (adminUserIds.isEmpty()) {
            log.warn("紧急告警：未找到管理员用户，跳过发送。modelId={}", modelId);
            return false;
        }
        String modelName = resolveModelName(modelId);
        String title = "模型低分紧急告警";
        String content = String.format("模型[%s]在24小时内收到%d条以上低分评价，请紧急关注。", modelName, URGENT_LOW_RATING_COUNT);

        MessageSendForm form = new MessageSendForm();
        form.setType("critical_alert");
        form.setTitle(title);
        form.setContent(content);
        form.setRecipientIds(adminUserIds);
        form.setBizModule("feedback");
        form.setBizId(modelId + ":urgent");
        form.setPriority(1);
        messageService.send(form);
        log.info("低分紧急告警已发送: modelId={}", modelId);
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
        form.setBizModule("feedback");
        form.setBizId(LocalDate.now() + ":severe");
        form.setPriority(2);
        messageService.send(form);
        log.info("低分严重告警已发送");
        return true;
    }

    private long countLowRatingsIn24Hours(String modelId) {
        if (modelId == null) {
            return 0;
        }
        return ratingMapper.selectCount(new LambdaQueryWrapper<SysRating>()
                .eq(SysRating::getModelId, modelId)
                .le(SysRating::getRating, LOW_RATING_THRESHOLD)
                .ge(SysRating::getCreateTime, LocalDateTime.now().minusHours(24)));
    }

    private String resolveModelName(String modelId) {
        if (modelId == null) {
            return "";
        }
        SysAiModel model = aiModelMapper.selectOne(new LambdaQueryWrapper<SysAiModel>()
                .eq(SysAiModel::getModelId, modelId)
                .last("LIMIT 1"));
        return model != null && model.getDisplayName() != null ? model.getDisplayName() : modelId;
    }

    private double calcDailyGlobalLowRatingRate() {
        LocalDateTime dayStart = LocalDate.now().atStartOfDay();
        long lowCount = ratingMapper.countLowRatingsSince(LOW_RATING_THRESHOLD, dayStart);
        long totalCount = ratingMapper.countRatingsSince(dayStart);
        if (totalCount == 0) {
            return 0.0;
        }
        return (double) lowCount * 100 / totalCount;
    }

    private List<Long> getAdminUserIds() {
        List<SysRole> roles = roleMapper.selectList(
                new LambdaQueryWrapper<SysRole>().in(SysRole::getCode, SystemConstants.ROOT_ROLE_CODE, "ADMIN"));
        if (roles.isEmpty()) {
            return Collections.emptyList();
        }
        List<Long> roleIds = roles.stream().map(SysRole::getId).toList();
        List<SysUserRole> userRoles =
                userRoleMapper.selectList(new LambdaQueryWrapper<SysUserRole>().in(SysUserRole::getRoleId, roleIds));
        return userRoles.stream().map(SysUserRole::getUserId).distinct().toList();
    }
}
