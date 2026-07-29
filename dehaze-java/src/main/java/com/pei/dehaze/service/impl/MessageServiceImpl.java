package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.config.WebSocketMessageRelay;
import com.pei.dehaze.mapper.SysMessageMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysMessage;
import com.pei.dehaze.model.entity.SysMessageTemplate;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.MessageSendForm;
import com.pei.dehaze.model.query.MessageQuery;
import com.pei.dehaze.model.query.MessageSearchQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.MessageService;
import com.pei.dehaze.service.MessageTemplateService;
import com.pei.dehaze.service.notify.MessagePushDispatcher;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
@RequiredArgsConstructor
public class MessageServiceImpl extends ServiceImpl<SysMessageMapper, SysMessage> implements MessageService {

    private static final Map<String, String> TYPE_LABELS = Map.of(
            "inbox", "站内信",
            "announcement", "系统公告",
            "business", "业务通知",
            "member", "会员通知",
            "alert", "告警通知",
            "critical_alert", "严重告警"
    );
    private static final Pattern VAR_PATTERN = Pattern.compile("\\{(\\w+)}");
    private static final String UNREAD_KEY_PREFIX = "msg:unread:";
    private static final DateTimeFormatter WS_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final MessageTemplateService messageTemplateService;
    private final MessagePushDispatcher pushDispatcher;
    private final StringRedisTemplate stringRedisTemplate;
    private final WebSocketMessageRelay webSocketMessageRelay;
    private final SysUserMapper userMapper;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public MessageSendResultVO send(MessageSendForm form) {
        if (CharSequenceUtil.isNotBlank(form.getBizModule()) && CharSequenceUtil.isNotBlank(form.getBizId())) {
            List<SysMessage> existing = this.list(new LambdaQueryWrapper<SysMessage>()
                    .eq(SysMessage::getBizModule, form.getBizModule())
                    .eq(SysMessage::getBizId, form.getBizId()));
            if (!existing.isEmpty()) {
                MessageSendResultVO vo = new MessageSendResultVO();
                vo.setMessageIds(existing.stream().map(SysMessage::getId).toList());
                return vo;
            }
        }

        String title = form.getTitle();
        String content = form.getContent();
        if (CharSequenceUtil.isNotBlank(form.getTemplateCode())) {
            SysMessageTemplate template = messageTemplateService.getByCode(form.getTemplateCode());
            if (template == null) {
                throw new BusinessException(ResultCode.MESSAGE_TEMPLATE_NOT_FOUND);
            }
            if (template.getStatus() != null && template.getStatus() == 0) {
                throw new BusinessException(ResultCode.TEMPLATE_DISABLED);
            }
            Map<String, String> variables = form.getVariables() != null ? form.getVariables() : Collections.emptyMap();
            validateTemplateVariables(template, variables);
            title = renderTemplate(template.getTitleTemplate(), variables);
            content = renderTemplate(template.getContentTemplate(), variables);
        } else {
            if (CharSequenceUtil.isBlank(title)) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "消息标题不能为空");
            }
            if (CharSequenceUtil.isBlank(content)) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "消息正文不能为空");
            }
        }

        String type = form.getType();
        Integer priority = form.getPriority() != null ? form.getPriority() : 2;
        LocalDateTime expiresAt = calcExpiresAt(type);

        List<Long> messageIds = new ArrayList<>();
        for (Long recipientId : form.getRecipientIds()) {
            SysMessage message = new SysMessage();
            message.setType(type);
            message.setTitle(title);
            message.setContent(content);
            message.setSenderType(1);
            message.setRecipientId(recipientId);
            message.setBizModule(form.getBizModule());
            message.setBizId(form.getBizId());
            message.setPriority(priority);
            message.setJumpUrl(form.getJumpUrl());
            message.setExtra(form.getExtra());
            message.setReadStatus(0);
            message.setDeleted(0);
            message.setExpiresAt(expiresAt);
            this.save(message);
            messageIds.add(message.getId());
            invalidateUnreadCache(recipientId);
            pushDispatcher.dispatch(message, recipientId);
            pushNewMessageEvent(message, recipientId);
        }

        MessageSendResultVO vo = new MessageSendResultVO();
        vo.setMessageIds(messageIds);
        return vo;
    }

    private void pushNewMessageEvent(SysMessage message, Long recipientId) {
        Map<String, Object> data = new HashMap<>();
        data.put("id", message.getId());
        data.put("type", message.getType());
        data.put("title", message.getTitle());
        data.put("priority", message.getPriority());
        data.put("createTime", message.getCreateTime() != null
                ? message.getCreateTime().format(WS_TIME_FORMATTER) : null);
        Map<String, Object> event = new HashMap<>();
        event.put("event", "new_message");
        event.put("data", data);
        webSocketMessageRelay.publishToUser(recipientId, WebSocketMessageRelay.DEST_MESSAGE, event);
    }

    private void validateTemplateVariables(SysMessageTemplate template, Map<String, String> variables) {
        String title = template.getTitleTemplate() != null ? template.getTitleTemplate() : "";
        String content = template.getContentTemplate() != null ? template.getContentTemplate() : "";
        Set<String> requiredVars = new HashSet<>();
        Matcher matcher = VAR_PATTERN.matcher(title + " " + content);
        while (matcher.find()) {
            requiredVars.add(matcher.group(1));
        }
        for (String varName : requiredVars) {
            if (!variables.containsKey(varName)) {
                throw new BusinessException(ResultCode.TEMPLATE_VAR_MISSING, "模板变量缺失: " + varName);
            }
        }
    }

    private String renderTemplate(String template, Map<String, String> variables) {
        if (CharSequenceUtil.isBlank(template)) {
            return template;
        }
        Matcher matcher = VAR_PATTERN.matcher(template);
        StringBuffer sb = new StringBuffer();
        while (matcher.find()) {
            String varName = matcher.group(1);
            String value = variables.getOrDefault(varName, "");
            matcher.appendReplacement(sb, Matcher.quoteReplacement(value));
        }
        matcher.appendTail(sb);
        return sb.toString();
    }

    private LocalDateTime calcExpiresAt(String type) {
        LocalDateTime now = LocalDateTime.now();
        return switch (type) {
            case "alert" -> now.plusDays(7);
            case "critical_alert" -> now.plusDays(90);
            default -> now.plusDays(30);
        };
    }

    @Override
    public Page<MessageVO> getPage(MessageQuery query) {
        Long userId = SecurityUtils.getUserId();
        Page<SysMessage> page = new Page<>(query.getPageNum(), query.getPageSize());
        this.page(page, new LambdaQueryWrapper<SysMessage>()
                .eq(SysMessage::getRecipientId, userId)
                .eq(query.getType() != null, SysMessage::getType, query.getType())
                .eq(query.getReadStatus() != null, SysMessage::getReadStatus, query.getReadStatus())
                .orderByAsc(SysMessage::getReadStatus)
                .orderByDesc(SysMessage::getCreateTime));

        Page<MessageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toMessageVO).toList());
        return result;
    }

    @Override
    public UnreadCountVO getUnreadCount() {
        Long userId = SecurityUtils.getUserId();
        String cacheKey = UNREAD_KEY_PREFIX + userId;
        String cached = stringRedisTemplate.opsForValue().get(cacheKey);
        long count;
        if (cached != null) {
            count = Long.parseLong(cached);
        } else {
            count = this.count(new LambdaQueryWrapper<SysMessage>()
                    .eq(SysMessage::getRecipientId, userId)
                    .eq(SysMessage::getReadStatus, 0));
            stringRedisTemplate.opsForValue().set(cacheKey, String.valueOf(count),
                    count == 0 ? Duration.ofMinutes(5) : Duration.ofHours(1));
        }
        UnreadCountVO vo = new UnreadCountVO();
        vo.setCount(count);
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public MessageDetailVO getDetail(Long id) {
        Long userId = SecurityUtils.getUserId();
        SysMessage message = this.getOne(new LambdaQueryWrapper<SysMessage>()
                .eq(SysMessage::getId, id)
                .eq(SysMessage::getRecipientId, userId));
        if (message == null) {
            throw new BusinessException(ResultCode.MESSAGE_NOT_FOUND);
        }

        if (message.getReadStatus() != null && message.getReadStatus() == 0) {
            LocalDateTime readTime = LocalDateTime.now();
            this.update(new LambdaUpdateWrapper<SysMessage>()
                    .eq(SysMessage::getId, id)
                    .eq(SysMessage::getReadStatus, 0)
                    .set(SysMessage::getReadStatus, 1)
                    .set(SysMessage::getReadTime, readTime));
            message.setReadStatus(1);
            message.setReadTime(readTime);
            invalidateUnreadCache(userId);
        }

        MessageDetailVO vo = new MessageDetailVO();
        copyToVO(message, vo);
        vo.setContent(message.getContent());
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void markRead(Long id) {
        Long userId = SecurityUtils.getUserId();
        int affected = this.getBaseMapper().update(null, new LambdaUpdateWrapper<SysMessage>()
                .eq(SysMessage::getId, id)
                .eq(SysMessage::getRecipientId, userId)
                .eq(SysMessage::getReadStatus, 0)
                .set(SysMessage::getReadStatus, 1)
                .set(SysMessage::getReadTime, LocalDateTime.now()));
        if (affected > 0) {
            invalidateUnreadCache(userId);
        }
    }

    @Override
    public ReadAllResultVO markAllRead(String type) {
        Long userId = SecurityUtils.getUserId();
        LambdaUpdateWrapper<SysMessage> wrapper = new LambdaUpdateWrapper<SysMessage>()
                .eq(SysMessage::getRecipientId, userId)
                .eq(SysMessage::getReadStatus, 0)
                .set(SysMessage::getReadStatus, 1)
                .set(SysMessage::getReadTime, LocalDateTime.now());
        if (CharSequenceUtil.isNotBlank(type)) {
            wrapper.eq(SysMessage::getType, type);
        }
        int affected = this.getBaseMapper().update(null, wrapper);
        if (affected > 0) {
            stringRedisTemplate.delete(UNREAD_KEY_PREFIX + userId);
        }
        ReadAllResultVO vo = new ReadAllResultVO();
        vo.setAffectedCount(affected);
        return vo;
    }

    @Override
    public void deleteByIds(String ids) {
        Long userId = SecurityUtils.getUserId();
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim)
                .filter(CharSequenceUtil::isNotBlank)
                .map(Long::parseLong)
                .toList();
        if (idList.isEmpty()) {
            return;
        }
        long unreadDeleted = this.count(new LambdaQueryWrapper<SysMessage>()
                .in(SysMessage::getId, idList)
                .eq(SysMessage::getRecipientId, userId)
                .eq(SysMessage::getReadStatus, 0));
        this.remove(new LambdaQueryWrapper<SysMessage>()
                .in(SysMessage::getId, idList)
                .eq(SysMessage::getRecipientId, userId));
        if (unreadDeleted > 0) {
            stringRedisTemplate.delete(UNREAD_KEY_PREFIX + userId);
        }
    }

    @Override
    public Page<MessageVO> search(MessageSearchQuery query) {
        Long userId = SecurityUtils.getUserId();
        String keyword = query.getKeyword();
        Page<SysMessage> page = new Page<>(query.getPageNum(), query.getPageSize());
        this.page(page, new LambdaQueryWrapper<SysMessage>()
                .eq(SysMessage::getRecipientId, userId)
                .and(w -> w.like(SysMessage::getTitle, keyword).or().like(SysMessage::getContent, keyword))
                .orderByAsc(SysMessage::getReadStatus)
                .orderByDesc(SysMessage::getCreateTime));

        Page<MessageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toMessageVO).toList());
        return result;
    }

    private MessageVO toMessageVO(SysMessage message) {
        MessageVO vo = new MessageVO();
        copyToVO(message, vo);
        vo.setSummary(summary(message.getContent()));
        return vo;
    }

    private String summary(String content) {
        if (content == null) {
            return null;
        }
        int[] codePoints = content.codePoints().toArray();
        if (codePoints.length <= 50) {
            return content;
        }
        return new String(codePoints, 0, 50);
    }

    private void copyToVO(SysMessage message, MessageVO vo) {
        vo.setId(message.getId());
        vo.setType(message.getType());
        vo.setTypeLabel(TYPE_LABELS.get(message.getType()));
        vo.setTitle(message.getTitle());
        vo.setPriority(message.getPriority());
        vo.setReadStatus(message.getReadStatus());
        vo.setSenderType(message.getSenderType());
        vo.setReadTime(message.getReadTime());
        vo.setJumpUrl(message.getJumpUrl());
        vo.setExtra(message.getExtra());
        vo.setCreateTime(message.getCreateTime());
    }

    private void invalidateUnreadCache(Long userId) {
        stringRedisTemplate.delete(UNREAD_KEY_PREFIX + userId);
    }

    @Override
    public void refreshUnreadCountCache() {
        List<SysUser> users = userMapper.selectList(new LambdaQueryWrapper<SysUser>()
                .eq(SysUser::getStatus, 1));
        if (users.isEmpty()) {
            log.info("未读数缓存刷新: 无活跃用户");
            return;
        }
        int refreshed = 0;
        for (SysUser user : users) {
            Long userId = user.getId();
            long count = this.count(new LambdaQueryWrapper<SysMessage>()
                    .eq(SysMessage::getRecipientId, userId)
                    .eq(SysMessage::getReadStatus, 0));
            stringRedisTemplate.opsForValue().set(UNREAD_KEY_PREFIX + userId, String.valueOf(count),
                    count == 0 ? Duration.ofMinutes(5) : Duration.ofHours(1));
            refreshed++;
        }
        log.info("未读数缓存刷新完成: 共刷新{}个用户", refreshed);
    }
}
