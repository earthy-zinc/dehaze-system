package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysKnowledgeBase;
import com.pei.dehaze.model.form.KbUpdateForm;
import com.pei.dehaze.model.form.TestSetForm;
import com.pei.dehaze.model.query.KbDocumentPageQuery;
import com.pei.dehaze.model.query.KbPageQuery;
import com.pei.dehaze.model.vo.KbChunkVO;
import com.pei.dehaze.model.vo.KbDocumentVO;
import com.pei.dehaze.model.vo.KbVO;
import com.pei.dehaze.model.vo.LowQualityChunkVO;
import com.pei.dehaze.model.vo.TestSetVO;

/**
 * AI 知识库查询与管理元数据（A 类）。
 *
 * <p>库的创建/删除与索引状态端点依赖 ES 索引生命周期，归 java-proxy 承担；
 * 文档版本更新与删除须驱动 python 文档处理流水线，同样走 java-proxy 转发。
 */
public interface AiKnowledgeBaseService extends IService<SysKnowledgeBase> {

    Page<KbVO> getPage(KbPageQuery query);

    KbVO getDetail(Long kbId);

    KbVO update(Long kbId, KbUpdateForm form);

    Page<KbDocumentVO> listDocuments(Long kbId, KbDocumentPageQuery query);

    KbDocumentVO getDocument(Long documentId);

    Page<KbChunkVO> listChunks(Long documentId, int page, int size);

    TestSetVO createTestSet(Long kbId, TestSetForm form);

    Page<TestSetVO> listTestSets(Long kbId, int page, int size);

    Page<LowQualityChunkVO> listLowQualityChunks(Long kbId, int page, int size);
}
