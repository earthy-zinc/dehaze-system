package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.model.entity.LoginLog;
import com.pei.dehaze.model.query.LoginLogQuery;
import com.pei.dehaze.model.vo.LoginLogVO;
import com.pei.dehaze.repository.LoginLogRepository;
import com.pei.dehaze.service.LoginLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class LoginLogServiceImpl implements LoginLogService {

    private static final DateTimeFormatter LOG_TIME_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    /**
     * 支持的日志时间格式（对齐 python auth_service._parse_log_time）
     */
    private static final List<DateTimeFormatter> LOG_TIME_PARSERS = List.of(
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd")
    );

    private final LoginLogRepository loginLogRepository;

    private final MongoTemplate mongoTemplate;

    @Async("datasetTaskExecutor")
    @Override
    public void recordLogin(Long userId, String username, String ip, int status, String message, String browser, String os, String deviceType) {
        try {
            LoginLog loginLog = new LoginLog();
            loginLog.setUserId(userId);
            loginLog.setUsername(username);
            loginLog.setIp(ip);
            loginLog.setBrowser(browser);
            loginLog.setOs(os);
            loginLog.setDeviceType(deviceType);
            loginLog.setStatus(status);
            loginLog.setMessage(message);
            loginLog.setCreateTime(LocalDateTime.now());
            loginLogRepository.save(loginLog);
        } catch (Exception e) {
            log.warn("写入登录日志失败: username={}, status={}", username, status, e);
        }
    }

    @Override
    public IPage<LoginLogVO> pageLoginLogs(LoginLogQuery query, Long restrictedUserId) {
        Criteria criteria = new Criteria();
        if (CharSequenceUtil.isNotBlank(query.getUsername())) {
            criteria.and("username").is(query.getUsername());
        }
        if (CharSequenceUtil.isNotBlank(query.getIp())) {
            criteria.and("ip").is(query.getIp());
        }
        if (query.getStatus() != null) {
            criteria.and("status").is(query.getStatus());
        }
        if (CharSequenceUtil.isNotBlank(query.getDeviceType())) {
            criteria.and("device_type").is(query.getDeviceType());
        }
        if (restrictedUserId != null) {
            criteria.and("user_id").is(restrictedUserId);
        }
        // 集合由 python 共写共读，条件键必须用实体映射后的 snake_case（见 LoginLog @Field）
        LocalDateTime startTime = parseLogTime(query.getStartTime());
        LocalDateTime endTime = parseLogTime(query.getEndTime());
        if (startTime != null || endTime != null) {
            // 同一字段的范围条件必须挂在同一 Criteria 链上：两次 and("create_time") 会让
            // Spring Data 在读键冲突时抛 InvalidMongoDbApiUsageException
            Criteria timeRange = criteria.and("create_time");
            if (startTime != null) {
                timeRange.gte(startTime);
            }
            if (endTime != null) {
                timeRange.lte(endTime);
            }
        }

        Query mongoQuery = new Query(criteria)
                .with(Sort.by(Sort.Direction.DESC, "create_time"))
                .skip((long) (query.getPageNum() - 1) * query.getPageSize())
                .limit(query.getPageSize());

        long total = mongoTemplate.count(new Query(criteria), LoginLog.class);
        List<LoginLog> logs = mongoTemplate.find(mongoQuery, LoginLog.class);

        Page<LoginLogVO> page = new Page<>(query.getPageNum(), query.getPageSize(), total);
        page.setRecords(logs.stream().map(this::toVO).toList());
        return page;
    }

    private LoginLogVO toVO(LoginLog entity) {
        LoginLogVO vo = new LoginLogVO();
        vo.setId(entity.getId());
        vo.setUserId(entity.getUserId());
        vo.setUsername(StrUtil.nullToEmpty(entity.getUsername()));
        vo.setIp(StrUtil.nullToEmpty(entity.getIp()));
        vo.setLocation(entity.getLocation());
        vo.setBrowser(entity.getBrowser());
        vo.setOs(entity.getOs());
        vo.setDeviceType(StrUtil.blankToDefault(entity.getDeviceType(), "web"));
        vo.setStatus(entity.getStatus());
        vo.setMessage(entity.getMessage());
        vo.setLoginTime(entity.getCreateTime() != null ? LOG_TIME_FORMAT.format(entity.getCreateTime()) : "");
        return vo;
    }

    private LocalDateTime parseLogTime(String value) {
        if (CharSequenceUtil.isBlank(value)) {
            return null;
        }
        for (DateTimeFormatter formatter : LOG_TIME_PARSERS) {
            try {
                return LocalDateTime.parse(value, formatter);
            } catch (Exception ignored) {
                // 尝试下一种格式
            }
        }
        return null;
    }
}
