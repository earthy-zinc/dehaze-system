package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysKnowledgeBaseMapper;
import com.pei.dehaze.mapper.SysKnowledgeChunkFeedbackMapper;
import com.pei.dehaze.mapper.SysKnowledgeChunkMapper;
import com.pei.dehaze.mapper.SysKnowledgeDocumentMapper;
import com.pei.dehaze.mapper.SysKnowledgeTestSetMapper;
import com.pei.dehaze.model.entity.SysKnowledgeBase;
import com.pei.dehaze.model.entity.SysKnowledgeChunk;
import com.pei.dehaze.model.entity.SysKnowledgeDocument;
import com.pei.dehaze.model.entity.SysKnowledgeTestSet;
import com.pei.dehaze.model.form.KbUpdateForm;
import com.pei.dehaze.model.form.TestSetForm;
import com.pei.dehaze.model.query.KbDocumentPageQuery;
import com.pei.dehaze.model.query.KbPageQuery;
import com.pei.dehaze.model.vo.KbChunkVO;
import com.pei.dehaze.model.vo.KbDocumentVO;
import com.pei.dehaze.model.vo.KbVO;
import com.pei.dehaze.model.vo.LowQualityChunkVO;
import com.pei.dehaze.model.vo.TestSetVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiKnowledgeBaseService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
@RequiredArgsConstructor
public class AiKnowledgeBaseServiceImpl extends ServiceImpl<SysKnowledgeBaseMapper, SysKnowledgeBase>
        implements AiKnowledgeBaseService {

    private static final long KB_LIST_TTL = 600L;
    private static final long KB_DETAIL_TTL = 1800L;
    /** 仅默认分页读写缓存，避免不同 size 互相污染 */
    private static final int DEFAULT_PAGE_SIZE = 10;
    private static final String ADMIN_LIST_CACHE_KEY = "kb:list:admin";

    private final SysKnowledgeDocumentMapper documentMapper;
    private final SysKnowledgeTestSetMapper testSetMapper;
    private final SysKnowledgeChunkMapper chunkMapper;
    private final SysKnowledgeChunkFeedbackMapper chunkFeedbackMapper;
    private final AiKbIndexClient kbIndexClient;
    private final StringRedisTemplate redis;
    private final ObjectMapper objectMapper;

    // ==================== 知识库 ====================

    @Override
    @Transactional(readOnly = true)
    public Page<KbVO> getPage(KbPageQuery query) {
        boolean admin = "admin".equals(query.getView());
        Long userId = SecurityUtils.getUserId();
        String cacheKey = admin ? ADMIN_LIST_CACHE_KEY : "kb:list:" + userId;
        boolean cacheable = CharSequenceUtil.isBlank(query.getKeyword())
                && query.getPageNum() == 1 && query.getPageSize() == DEFAULT_PAGE_SIZE;
        if (cacheable) {
            Page<KbVO> cached = readPageCache(cacheKey);
            if (cached != null) {
                return cached;
            }
        }
        LambdaQueryWrapper<SysKnowledgeBase> wrapper = new LambdaQueryWrapper<SysKnowledgeBase>()
                .like(CharSequenceUtil.isNotBlank(query.getKeyword()), SysKnowledgeBase::getName, query.getKeyword())
                .orderByDesc(SysKnowledgeBase::getId);
        if (!admin) {
            // 可见性：公开库全员可见 + 本人私有库
            wrapper.and(w -> w.eq(SysKnowledgeBase::getVisibility, "public")
                    .or()
                    .eq(SysKnowledgeBase::getCreateBy, userId));
        }
        Page<SysKnowledgeBase> page = this.page(new Page<>(query.getPageNum(), query.getPageSize()), wrapper);
        Page<KbVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toVO).toList());
        if (cacheable) {
            writePageCache(cacheKey, result);
        }
        return result;
    }

    @Override
    @Transactional(readOnly = true)
    public KbVO getDetail(Long kbId) {
        SysKnowledgeBase kb = getOrRaise(kbId);
        checkReadable(kb);
        String raw = redis.opsForValue().get(detailCacheKey(kbId));
        if (CharSequenceUtil.isNotBlank(raw)) {
            KbVO cached = readJson(raw, KbVO.class);
            if (cached != null) {
                return cached;
            }
            redis.delete(detailCacheKey(kbId));
        }
        KbVO vo = toVO(kb);
        writeJson(detailCacheKey(kbId), vo, KB_DETAIL_TTL);
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public KbVO update(Long kbId, KbUpdateForm form) {
        SysKnowledgeBase kb = getOrRaise(kbId);
        checkManagePermission(kb);
        if (CharSequenceUtil.isNotBlank(form.getEmbeddingModel())
                || CharSequenceUtil.isNotBlank(form.getChunkingStrategy())) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "创建后不可修改 embedding 模型或分块策略");
        }
        if (form.getName() != null && !form.getName().equals(kb.getName())) {
            if (getByNameAndOwner(form.getName(), kb.getCreateBy()) != null) {
                throw new BusinessException(ResultCode.BUSINESS_ERROR, "知识库名称已存在");
            }
            kb.setName(form.getName());
        }
        if (form.getDescription() != null) {
            kb.setDescription(form.getDescription());
        }
        if (form.getSearchStrategy() != null) {
            kb.setSearchStrategy(form.getSearchStrategy());
        }
        if (form.getHybridWeight() != null) {
            kb.setHybridWeight(form.getHybridWeight());
        }
        if (form.getTopK() != null) {
            kb.setTopK(form.getTopK());
        }
        if (form.getScoreThreshold() != null) {
            kb.setScoreThreshold(form.getScoreThreshold());
        }
        if (form.getEnableRerank() != null) {
            kb.setEnableRerank(Boolean.TRUE.equals(form.getEnableRerank()) ? 1 : 0);
        }
        if (form.getRerankModel() != null) {
            kb.setRerankModel(form.getRerankModel());
        }
        this.updateById(kb);
        clearCache(SecurityUtils.getUserId(), kbId, false);
        clearDetailCache(kbId);
        return toVO(kb);
    }

    // ==================== 文档 ====================

    @Override
    @Transactional(readOnly = true)
    public Page<KbDocumentVO> listDocuments(Long kbId, KbDocumentPageQuery query) {
        SysKnowledgeBase kb = getOrRaise(kbId);
        if ("private".equals(kb.getVisibility()) && !kb.getCreateBy().equals(SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权查看他人私有知识库");
        }
        LambdaQueryWrapper<SysKnowledgeDocument> wrapper = new LambdaQueryWrapper<SysKnowledgeDocument>()
                .eq(SysKnowledgeDocument::getKnowledgeBaseId, kbId)
                .eq(CharSequenceUtil.isNotBlank(query.getProcessingStatus()),
                        SysKnowledgeDocument::getProcessingStatus, query.getProcessingStatus())
                .orderByDesc(SysKnowledgeDocument::getId);
        Page<SysKnowledgeDocument> page = documentMapper.selectPage(
                new Page<>(query.getPageNum(), query.getPageSize()), wrapper);
        Page<KbDocumentVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        // 列表不返回大字段 content，详情单独返回（避免列表载荷过大）
        result.setRecords(page.getRecords().stream().map(doc -> toDocumentVO(doc, false)).toList());
        return result;
    }

    @Override
    @Transactional(readOnly = true)
    public KbDocumentVO getDocument(Long documentId) {
        SysKnowledgeDocument doc = getDocumentOrRaise(documentId);
        checkDocumentReadable(doc);
        return toDocumentVO(doc, true);
    }

    @Override
    @Transactional(readOnly = true)
    public Page<KbChunkVO> listChunks(Long documentId, int page, int size) {
        SysKnowledgeDocument doc = getDocumentOrRaise(documentId);
        checkDocumentReadable(doc);
        Page<SysKnowledgeChunk> chunkPage = chunkMapper.selectPage(new Page<>(page, size),
                new LambdaQueryWrapper<SysKnowledgeChunk>()
                        .eq(SysKnowledgeChunk::getDocumentId, documentId)
                        .orderByAsc(SysKnowledgeChunk::getChunkIndex));
        Page<KbChunkVO> result = new Page<>(chunkPage.getCurrent(), chunkPage.getSize(), chunkPage.getTotal());
        result.setRecords(chunkPage.getRecords().stream().map(this::toChunkVO).toList());
        return result;
    }

    // ==================== 召回测试集 / 低质量片段 ====================

    @Override
    @Transactional(rollbackFor = Exception.class)
    public TestSetVO createTestSet(Long kbId, TestSetForm form) {
        getOrRaise(kbId);
        SysKnowledgeTestSet entity = new SysKnowledgeTestSet();
        entity.setKnowledgeBaseId(kbId);
        entity.setQuestion(form.getQuestion());
        entity.setExpectedChunkIds(writeJsonString(form.getExpectedChunkIds()));
        testSetMapper.insert(entity);
        return toTestSetVO(entity);
    }

    @Override
    @Transactional(readOnly = true)
    public Page<TestSetVO> listTestSets(Long kbId, int page, int size) {
        getOrRaise(kbId);
        Page<SysKnowledgeTestSet> result = testSetMapper.selectPage(new Page<>(page, size),
                new LambdaQueryWrapper<SysKnowledgeTestSet>()
                        .eq(SysKnowledgeTestSet::getKnowledgeBaseId, kbId)
                        .orderByDesc(SysKnowledgeTestSet::getId));
        Page<TestSetVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());
        voPage.setRecords(result.getRecords().stream().map(this::toTestSetVO).toList());
        return voPage;
    }

    @Override
    @Transactional(readOnly = true)
    public Page<LowQualityChunkVO> listLowQualityChunks(Long kbId, int page, int size) {
        getOrRaise(kbId);
        long total = chunkFeedbackMapper.countLowQualityByKb(kbId);
        int offset = (page - 1) * size;
        List<Map<String, Object>> rows = chunkFeedbackMapper.listLowQualityByKb(kbId, size, offset);
        List<LowQualityChunkVO> items = new ArrayList<>(rows.size());
        for (Map<String, Object> row : rows) {
            LowQualityChunkVO vo = new LowQualityChunkVO();
            vo.setChunkId(longValue(row.get("chunkId")));
            vo.setContent(row.get("content") == null ? null : String.valueOf(row.get("content")));
            vo.setDocumentId(longValue(row.get("documentId")));
            vo.setThumbsDownCount(intValue(row.get("thumbsDownCount")));
            items.add(vo);
        }
        Page<LowQualityChunkVO> result = new Page<>(page, size, total);
        result.setRecords(items);
        return result;
    }

    // ==================== 内部实现 ====================

    private SysKnowledgeBase getOrRaise(Long kbId) {
        SysKnowledgeBase kb = this.getById(kbId);
        if (kb == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "知识库不存在");
        }
        return kb;
    }

    private SysKnowledgeDocument getDocumentOrRaise(Long documentId) {
        SysKnowledgeDocument doc = documentMapper.selectById(documentId);
        if (doc == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文档不存在");
        }
        return doc;
    }

    /** 管理权限：私有库仅 owner，公开库仅管理员 */
    private void checkManagePermission(SysKnowledgeBase kb) {
        if ("public".equals(kb.getVisibility())) {
            if (!SecurityUtils.isAdmin()) {
                throw new BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "普通用户不能管理公共知识库");
            }
            return;
        }
        if (!kb.getCreateBy().equals(SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权操作他人私有知识库");
        }
    }

    private void checkReadable(SysKnowledgeBase kb) {
        if ("private".equals(kb.getVisibility())
                && !kb.getCreateBy().equals(SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权查看他人私有知识库");
        }
    }

    private void checkDocumentReadable(SysKnowledgeDocument doc) {
        SysKnowledgeBase kb = getOrRaise(doc.getKnowledgeBaseId());
        if ("private".equals(kb.getVisibility())
                && !kb.getCreateBy().equals(SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "无权查看他人私有知识库的文档");
        }
    }

    private SysKnowledgeBase getByNameAndOwner(String name, Long ownerId) {
        return this.getOne(new LambdaQueryWrapper<SysKnowledgeBase>()
                .eq(SysKnowledgeBase::getName, name)
                .eq(SysKnowledgeBase::getCreateBy, ownerId)
                .orderByAsc(SysKnowledgeBase::getId), false);
    }

    /** 知识库变更后失效相关缓存：用户列表 + 详情 + 管理端全局列表（含新库） */
    private void clearCache(Long userId, Long kbId, boolean withAdminList) {
        List<String> keys = new ArrayList<>();
        if (userId != null) {
            keys.add("kb:list:" + userId);
        }
        if (kbId != null) {
            keys.add(detailCacheKey(kbId));
        }
        if (withAdminList) {
            keys.add(ADMIN_LIST_CACHE_KEY);
        }
        redis.delete(keys);
    }

    private void clearDetailCache(Long kbId) {
        redis.delete(detailCacheKey(kbId));
    }

    private static String detailCacheKey(Long kbId) {
        return "kb:detail:" + kbId;
    }

    /** 缓存为 {list,total} 结构（与 python 同格式），反序列化失败时删键回源 */
    private Page<KbVO> readPageCache(String cacheKey) {
        String raw = redis.opsForValue().get(cacheKey);
        if (CharSequenceUtil.isBlank(raw)) {
            return null;
        }
        try {
            Map<String, Object> node = objectMapper.readValue(raw, new TypeReference<Map<String, Object>>() {
            });
            List<KbVO> list = objectMapper.convertValue(node.get("list"), new TypeReference<List<KbVO>>() {
            });
            long total = node.get("total") instanceof Number number ? number.longValue() : 0L;
            Page<KbVO> page = new Page<>(1, DEFAULT_PAGE_SIZE, total);
            page.setRecords(list == null ? List.of() : list);
            return page;
        } catch (Exception e) {
            log.warn("知识库列表缓存[{}]解析失败，删除后回源: {}", cacheKey, e.getMessage());
            redis.delete(cacheKey);
            return null;
        }
    }

    private void writePageCache(String cacheKey, Page<KbVO> page) {
        Map<String, Object> node = new LinkedHashMap<>();
        node.put("list", page.getRecords());
        node.put("total", page.getTotal());
        writeJson(cacheKey, node, KB_LIST_TTL);
    }

    private void writeJson(String key, Object value, long ttl) {
        try {
            redis.opsForValue().set(key, objectMapper.writeValueAsString(value), ttl, TimeUnit.SECONDS);
        } catch (Exception e) {
            log.warn("知识库缓存[{}]写入失败: {}", key, e.getMessage());
        }
    }

    private <T> T readJson(String raw, Class<T> type) {
        try {
            return objectMapper.readValue(raw, type);
        } catch (Exception e) {
            return null;
        }
    }

    private String writeJsonString(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "测试集期望分块格式非法");
        }
    }

    private KbVO toVO(SysKnowledgeBase kb) {
        KbVO vo = new KbVO();
        vo.setId(kb.getId());
        vo.setName(kb.getName());
        vo.setDescription(kb.getDescription());
        vo.setVisibility(kb.getVisibility());
        vo.setEmbeddingProvider(kb.getEmbeddingProvider());
        vo.setEmbeddingModel(kb.getEmbeddingModel());
        vo.setChunkingStrategy(kb.getChunkingStrategy());
        vo.setChunkSize(kb.getChunkSize());
        vo.setChunkOverlap(kb.getChunkOverlap());
        vo.setSearchStrategy(kb.getSearchStrategy());
        vo.setHybridWeight(kb.getHybridWeight());
        vo.setTopK(kb.getTopK());
        vo.setScoreThreshold(kb.getScoreThreshold());
        vo.setEnableRerank(kb.getEnableRerank());
        vo.setRerankModel(kb.getRerankModel());
        vo.setDocumentCount(kb.getDocumentCount());
        vo.setChunkCount(kb.getChunkCount());
        vo.setTotalTokens(kb.getTotalTokens());
        vo.setStatus(kb.getStatus());
        vo.setCreateBy(kb.getCreateBy());
        vo.setCreateTime(kb.getCreateTime());
        vo.setUpdateTime(kb.getUpdateTime());
        return vo;
    }

    private KbDocumentVO toDocumentVO(SysKnowledgeDocument doc, boolean withContent) {
        KbDocumentVO vo = new KbDocumentVO();
        vo.setId(doc.getId());
        vo.setKnowledgeBaseId(doc.getKnowledgeBaseId());
        vo.setFileId(doc.getFileId());
        vo.setTitle(doc.getTitle());
        vo.setSource(doc.getSource());
        vo.setVersion(doc.getVersion());
        vo.setParsingStrategy(doc.getParsingStrategy());
        vo.setChunkCount(doc.getChunkCount());
        vo.setTotalTokens(doc.getTotalTokens());
        vo.setProcessingStatus(doc.getProcessingStatus());
        vo.setError(doc.getError());
        vo.setCreateTime(doc.getCreateTime());
        vo.setUpdateTime(doc.getUpdateTime());
        if (withContent) {
            vo.setContent(doc.getContent());
            vo.setRawContent(doc.getRawContent());
        }
        return vo;
    }

    /** 分块元数据在 MySQL 为 JSON 列，序列化回对象以对齐 python KnowledgeChunkVO.metadata */
    private KbChunkVO toChunkVO(SysKnowledgeChunk chunk) {
        KbChunkVO vo = new KbChunkVO();
        vo.setId(chunk.getId());
        vo.setDocumentId(chunk.getDocumentId());
        vo.setChunkIndex(chunk.getChunkIndex());
        vo.setContent(chunk.getContent());
        vo.setTokenCount(chunk.getTokenCount());
        vo.setCreateTime(chunk.getCreateTime());
        if (CharSequenceUtil.isNotBlank(chunk.getMetadata())) {
            vo.setMetadata(readJson(chunk.getMetadata(), Object.class));
        }
        return vo;
    }

    private TestSetVO toTestSetVO(SysKnowledgeTestSet entity) {
        TestSetVO vo = new TestSetVO();
        vo.setId(entity.getId());
        vo.setKnowledgeBaseId(entity.getKnowledgeBaseId());
        vo.setQuestion(entity.getQuestion());
        List<Long> chunkIds = new ArrayList<>();
        if (CharSequenceUtil.isNotBlank(entity.getExpectedChunkIds())) {
            try {
                List<Object> raw = objectMapper.readValue(entity.getExpectedChunkIds(),
                        new TypeReference<List<Object>>() {
                        });
                for (Object item : raw) {
                    chunkIds.add(item instanceof Number number ? number.longValue()
                            : Long.parseLong(String.valueOf(item)));
                }
            } catch (Exception e) {
                log.warn("测试集期望分块解析失败 id={}: {}", entity.getId(), e.getMessage());
            }
        }
        vo.setExpectedChunkIds(chunkIds);
        vo.setCreateTime(entity.getCreateTime());
        return vo;
    }

    private static Long longValue(Object value) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        return value == null ? null : Long.parseLong(String.valueOf(value));
    }

    private static Integer intValue(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        return value == null ? 0 : Integer.parseInt(String.valueOf(value));
    }
}
