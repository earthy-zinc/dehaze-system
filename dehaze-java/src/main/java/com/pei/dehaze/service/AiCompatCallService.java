package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.util.AiDateTimeUtils;
import com.pei.dehaze.model.entity.AiApiCallLog;
import com.pei.dehaze.model.query.AiCompatCallQuery;
import com.pei.dehaze.model.vo.AiCompatCallVO;
import com.pei.dehaze.security.util.SecurityUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;

/**
 * AI 兼容 API 调用审计查询（仅本人数据），对齐 dehaze-python {@code ai_compat_call.py}
 * 与 dehaze-go {@code CompatAuditService}：user_id 强制过滤、时间非法按无过滤处理、
 * 分页参数为 page/size、create_time 倒序。
 *
 * <p>时间口径：python 写入 {@code datetime.now(UTC)}，查询入参按无时区字符串解析后被
 * pymongo 视作 UTC，故 java 同样把 `yyyy-MM-dd HH:mm:ss` 解析为 UTC 时刻，返回时再按
 * UTC 墙钟展示，保持与 python 端同一时刻。
 *
 * @author dehaze
 */
@Service
@RequiredArgsConstructor
public class AiCompatCallService {

    private final MongoTemplate mongoTemplate;

    public IPage<AiCompatCallVO> listCalls(AiCompatCallQuery query) {
        Criteria criteria = Criteria.where("user_id").is(SecurityUtils.getUserId());
        if (query.getKeyId() != null) {
            criteria.and("key_id").is(query.getKeyId());
        }
        if (StringUtils.hasText(query.getModel())) {
            criteria.and("model").is(query.getModel());
        }
        Instant startTime = toUtcInstant(query.getStartTime());
        Instant endTime = toUtcInstant(query.getEndTime());
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
                .skip((long) (query.getPage() - 1) * query.getSize())
                .limit(query.getSize());
        long total = mongoTemplate.count(new Query(criteria), AiApiCallLog.class);
        List<AiCompatCallVO> records = new ArrayList<>();
        for (AiApiCallLog log : mongoTemplate.find(mongoQuery, AiApiCallLog.class)) {
            records.add(toVO(log));
        }
        Page<AiCompatCallVO> page = new Page<>(query.getPage(), query.getSize(), total);
        page.setRecords(records);
        return page;
    }

    private AiCompatCallVO toVO(AiApiCallLog log) {
        AiCompatCallVO vo = new AiCompatCallVO();
        vo.setId(log.getId());
        vo.setKeyId(log.getKeyId());
        vo.setKeyPrefix(log.getKeyPrefix());
        vo.setConversationId(log.getConversationId());
        vo.setModel(log.getModel());
        vo.setEndpoint(log.getEndpoint());
        vo.setProtocol(log.getProtocol());
        vo.setIsStream(log.getIsStream());
        vo.setInputTokens(log.getInputTokens());
        vo.setOutputTokens(log.getOutputTokens());
        vo.setCredits(log.getCredits());
        vo.setStatusCode(log.getStatusCode());
        vo.setDurationMs(log.getDurationMs());
        vo.setClientIp(log.getClientIp());
        vo.setRequestId(log.getRequestId());
        vo.setErrorMsg(log.getErrorMsg());
        vo.setCreateTime(log.getCreateTime() == null ? null
                : LocalDateTime.ofInstant(log.getCreateTime(), ZoneOffset.UTC));
        return vo;
    }

    private static Instant toUtcInstant(String value) {
        LocalDateTime parsed = AiDateTimeUtils.parseOrNull(value);
        return parsed == null ? null : parsed.toInstant(ZoneOffset.UTC);
    }
}
