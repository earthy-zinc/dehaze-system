package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.form.KbUpdateForm;
import com.pei.dehaze.model.form.TestSetForm;
import com.pei.dehaze.model.query.KbDocumentPageQuery;
import com.pei.dehaze.model.query.KbPageQuery;
import com.pei.dehaze.model.query.PageParamQuery;
import com.pei.dehaze.model.vo.KbChunkVO;
import com.pei.dehaze.model.vo.KbDocumentVO;
import com.pei.dehaze.model.vo.KbVO;
import com.pei.dehaze.model.vo.LowQualityChunkVO;
import com.pei.dehaze.model.vo.TestSetVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiKnowledgeBaseService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * AI 知识库（查询与管理元数据）。
 *
 * <p>依赖 python 处理管线与 ES 索引生命周期的端点（库创建/删除、索引状态、文档版本更新、
 * 文档删除）归 java-proxy 转发——文档更新/删除须触发 python 的文档处理流水线，java 原生实现
 * 无法驱动异步重解析，会使 processingStatus 永久停留在 pending。
 */
@Tag(name = "AI知识库")
@RestController
@RequestMapping("/api/v1/kb")
@RequiredArgsConstructor
public class AiKnowledgeBaseController {

    private static final String KB_AUDIT_PERMISSION = "kb:audit";

    private final AiKnowledgeBaseService aiKnowledgeBaseService;

    @Operation(summary = "知识库列表")
    @GetMapping
    public PageResult<KbVO> listKnowledgeBases(@Valid @ParameterObject KbPageQuery query) {
        // view=admin 管理端视角：仅 kb:audit 可看全量含私有库
        if ("admin".equals(query.getView()) && !SecurityUtils.isRoot()
                && !SecurityUtils.getPerms().contains(KB_AUDIT_PERMISSION)) {
            throw new BusinessException(ResultCode.ACCESS_UNAUTHORIZED);
        }
        return PageResult.success(aiKnowledgeBaseService.getPage(query));
    }

    @Operation(summary = "知识库详情")
    @GetMapping("/{kbId}")
    public Result<KbVO> getKnowledgeBase(@PathVariable Long kbId) {
        return Result.success(aiKnowledgeBaseService.getDetail(kbId));
    }

    @Operation(summary = "编辑知识库")
    @PutMapping("/{kbId}")
    @PreAuthorize("@ss.hasPerm('kb:manage')")
    public Result<KbVO> updateKnowledgeBase(@PathVariable Long kbId,
                                            @RequestBody @Valid KbUpdateForm form) {
        return Result.success(aiKnowledgeBaseService.update(kbId, form));
    }

    @Operation(summary = "知识库文档列表")
    @GetMapping("/{kbId}/documents")
    public PageResult<KbDocumentVO> listDocuments(@PathVariable Long kbId,
                                                  @Valid @ParameterObject KbDocumentPageQuery query) {
        return PageResult.success(aiKnowledgeBaseService.listDocuments(kbId, query));
    }

    @Operation(summary = "文档详情")
    @GetMapping("/documents/{documentId}")
    public Result<KbDocumentVO> getDocument(@PathVariable Long documentId) {
        return Result.success(aiKnowledgeBaseService.getDocument(documentId));
    }

    @Operation(summary = "文档分块列表")
    @GetMapping("/documents/{documentId}/chunks")
    public PageResult<KbChunkVO> listDocumentChunks(@PathVariable Long documentId,
                                                    @Valid @ParameterObject PageParamQuery query) {
        Page<KbChunkVO> page = aiKnowledgeBaseService.listChunks(
                documentId, query.getPageNum(), query.getPageSize());
        return PageResult.success(page);
    }

    @Operation(summary = "创建召回测试集")
    @PostMapping("/{kbId}/retrieve/test-sets")
    @PreAuthorize("@ss.hasPerm('kb:audit')")
    public Result<TestSetVO> createTestSet(@PathVariable Long kbId,
                                           @RequestBody @Valid TestSetForm form) {
        return Result.success(aiKnowledgeBaseService.createTestSet(kbId, form));
    }

    @Operation(summary = "召回测试集列表")
    @GetMapping("/{kbId}/retrieve/test-sets")
    @PreAuthorize("@ss.hasPerm('kb:audit')")
    public PageResult<TestSetVO> listTestSets(@PathVariable Long kbId,
                                              @Valid @ParameterObject PageParamQuery query) {
        Page<TestSetVO> page = aiKnowledgeBaseService.listTestSets(
                kbId, query.getPageNum(), query.getPageSize());
        return PageResult.success(page);
    }

    @Operation(summary = "低质量片段列表")
    @GetMapping("/{kbId}/chunks/low-quality")
    @PreAuthorize("@ss.hasPerm('kb:audit')")
    public PageResult<LowQualityChunkVO> listLowQualityChunks(@PathVariable Long kbId,
                                                              @Valid @ParameterObject PageParamQuery query) {
        Page<LowQualityChunkVO> page = aiKnowledgeBaseService.listLowQualityChunks(
                kbId, query.getPageNum(), query.getPageSize());
        return PageResult.success(page);
    }
}
