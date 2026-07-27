package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAnnouncementMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysAnnouncement;
import com.pei.dehaze.model.entity.SysMessage;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.AnnouncementForm;
import com.pei.dehaze.model.query.AnnouncementQuery;
import com.pei.dehaze.model.vo.AnnouncementDetailVO;
import com.pei.dehaze.model.vo.AnnouncementSendResultVO;
import com.pei.dehaze.model.vo.AnnouncementVO;
import com.pei.dehaze.service.AnnouncementService;
import com.pei.dehaze.service.MessageService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;

@Service
@RequiredArgsConstructor
public class AnnouncementServiceImpl extends ServiceImpl<SysAnnouncementMapper, SysAnnouncement> implements AnnouncementService {

    private static final Map<String, String> TYPE_LABELS = Map.of(
            "maintenance", "系统维护",
            "feature", "功能更新",
            "activity", "活动通知",
            "operation", "运营公告"
    );
    private static final Map<String, String> SCOPE_LABELS = Map.of(
            "all", "全体用户",
            "level", "按会员等级",
            "tag", "按用户标签",
            "specified", "指定用户"
    );
    private static final Map<Integer, String> STATUS_LABELS = Map.of(
            1, "草稿",
            2, "待发送",
            3, "已发送",
            4, "已取消"
    );
    private static final Map<Integer, String> IMPORTANCE_LABELS = Map.of(
            1, "普通",
            2, "重要"
    );

    private final SysUserMapper sysUserMapper;
    private final MessageService messageService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Long create(AnnouncementForm form) {
        SysAnnouncement entity = new SysAnnouncement();
        entity.setTitle(form.getTitle());
        entity.setContent(form.getContent());
        entity.setType(form.getType());
        entity.setImportance(form.getImportance());
        entity.setTargetScope(form.getTargetScope());
        entity.setTargetParams(form.getTargetParams() != null ? JSONUtil.toJsonStr(form.getTargetParams()) : null);
        entity.setSendTime(form.getSendTime());
        entity.setExpireTime(form.getExpireTime());
        entity.setSentCount(0);

        if (form.getSendTime() != null && form.getSendTime().isAfter(LocalDateTime.now())) {
            entity.setStatus(2);
        } else {
            entity.setStatus(1);
        }
        this.save(entity);
        return entity.getId();
    }

    @Override
    public Page<AnnouncementVO> getPage(AnnouncementQuery query) {
        Page<SysAnnouncement> page = new Page<>(query.getPageNum(), query.getPageSize());
        this.page(page, new LambdaQueryWrapper<SysAnnouncement>()
                .like(CharSequenceUtil.isNotBlank(query.getTitle()), SysAnnouncement::getTitle, query.getTitle())
                .eq(CharSequenceUtil.isNotBlank(query.getType()), SysAnnouncement::getType, query.getType())
                .eq(query.getStatus() != null, SysAnnouncement::getStatus, query.getStatus())
                .orderByDesc(SysAnnouncement::getCreateTime));

        Page<AnnouncementVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toVO).toList());
        return result;
    }

    @Override
    public AnnouncementDetailVO getDetail(Long id) {
        SysAnnouncement entity = this.getById(id);
        if (entity == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        AnnouncementDetailVO vo = new AnnouncementDetailVO();
        copyToVO(entity, vo);
        vo.setContent(entity.getContent());
        vo.setImportanceLabel(IMPORTANCE_LABELS.get(entity.getImportance()));
        vo.setUpdateTime(entity.getUpdateTime());
        if (CharSequenceUtil.isNotBlank(entity.getTargetParams())) {
            vo.setTargetParams(JSONUtil.parseObj(entity.getTargetParams()).toBean(Map.class));
        }
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(Long id, AnnouncementForm form) {
        SysAnnouncement entity = this.getById(id);
        if (entity == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        if (entity.getStatus() == 3) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW);
        }
        if (form.getTitle() != null) {
            entity.setTitle(form.getTitle());
        }
        if (form.getContent() != null) {
            entity.setContent(form.getContent());
        }
        if (form.getType() != null) {
            entity.setType(form.getType());
        }
        if (form.getImportance() != null) {
            entity.setImportance(form.getImportance());
        }
        if (form.getTargetScope() != null) {
            entity.setTargetScope(form.getTargetScope());
        }
        if (form.getTargetParams() != null) {
            entity.setTargetParams(JSONUtil.toJsonStr(form.getTargetParams()));
        }
        if (form.getSendTime() != null) {
            entity.setSendTime(form.getSendTime());
        }
        if (form.getExpireTime() != null) {
            entity.setExpireTime(form.getExpireTime());
        }
        this.updateById(entity);
    }

    @Override
    public void delete(Long id) {
        SysAnnouncement entity = this.getById(id);
        if (entity == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        this.removeById(id);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public AnnouncementSendResultVO send(Long id) {
        SysAnnouncement entity = this.getById(id);
        if (entity == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

        List<Long> recipientIds = resolveTargetUserIds(entity);
        LocalDateTime expiresAt = LocalDateTime.now().plusDays(30);
        String bizId = String.valueOf(id);

        List<SysMessage> messages = new ArrayList<>();
        for (Long recipientId : recipientIds) {
            SysMessage message = new SysMessage();
            message.setType("announcement");
            message.setTitle(entity.getTitle());
            message.setContent(entity.getContent());
            message.setSenderType(2);
            message.setRecipientId(recipientId);
            message.setBizModule("system");
            message.setBizId(bizId);
            message.setPriority(entity.getImportance() != null && entity.getImportance() == 2 ? 3 : 2);
            message.setReadStatus(0);
            message.setDeleted(0);
            message.setExpiresAt(expiresAt);
            messages.add(message);
        }
        if (!messages.isEmpty()) {
            messageService.saveBatch(messages);
        }

        entity.setStatus(3);
        entity.setSentCount(recipientIds.size());
        entity.setSendTime(LocalDateTime.now());
        this.updateById(entity);

        AnnouncementSendResultVO vo = new AnnouncementSendResultVO();
        vo.setSentCount(recipientIds.size());
        return vo;
    }

    @Override
    public void cancel(Long id) {
        SysAnnouncement entity = this.getById(id);
        if (entity == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        if (entity.getStatus() != 2) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW);
        }
        entity.setStatus(4);
        this.updateById(entity);
    }

    private List<Long> resolveTargetUserIds(SysAnnouncement entity) {
        String scope = entity.getTargetScope();
        Map<String, Object> params = CharSequenceUtil.isNotBlank(entity.getTargetParams())
                ? JSONUtil.parseObj(entity.getTargetParams()).toBean(Map.class) : Collections.emptyMap();
        return switch (scope) {
            case "all" -> sysUserMapper.selectList(new LambdaQueryWrapper<SysUser>()
                    .select(SysUser::getId))
                    .stream().map(SysUser::getId).toList();
            case "level" -> {
                Object level = params.get("level");
                if (level == null) {
                    yield Collections.emptyList();
                }
                yield this.baseMapper.selectUserIdsByLevel(Integer.valueOf(level.toString()));
            }
            case "specified" -> {
                Object userIds = params.get("userIds");
                if (userIds instanceof List<?> list) {
                    yield list.stream().filter(Objects::nonNull).map(o -> Long.valueOf(o.toString())).toList();
                }
                yield Collections.emptyList();
            }
            default -> Collections.emptyList();
        };
    }

    private AnnouncementVO toVO(SysAnnouncement entity) {
        AnnouncementVO vo = new AnnouncementVO();
        copyToVO(entity, vo);
        vo.setImportanceLabel(IMPORTANCE_LABELS.get(entity.getImportance()));
        return vo;
    }

    private void copyToVO(SysAnnouncement entity, AnnouncementVO vo) {
        vo.setId(entity.getId());
        vo.setTitle(entity.getTitle());
        vo.setContent(entity.getContent());
        vo.setType(entity.getType());
        vo.setTypeLabel(TYPE_LABELS.get(entity.getType()));
        vo.setImportance(entity.getImportance());
        vo.setTargetScope(entity.getTargetScope());
        vo.setTargetScopeLabel(SCOPE_LABELS.get(entity.getTargetScope()));
        vo.setStatus(entity.getStatus());
        vo.setStatusLabel(STATUS_LABELS.get(entity.getStatus()));
        vo.setSendTime(entity.getSendTime());
        vo.setExpireTime(entity.getExpireTime());
        vo.setSentCount(entity.getSentCount());
        vo.setCreateTime(entity.getCreateTime());
        vo.setCreateBy(entity.getCreateBy());
    }
}
