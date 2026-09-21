package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiArtifactMapper;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.mapper.SysFileMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAiArtifact;
import com.pei.dehaze.model.entity.SysAiConversation;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.vo.AiArtifactVO;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * AI 中间产物服务。
 *
 * <p>对齐 dehaze-python {@code ai_artifact_service}：产物归属随会话校验；图片 URL 运行时按
 * ref 链路（sys_file / sys_pred_log / sys_eval_log）拼接，遵循"URL 永不落库"。
 *
 * @author dehaze
 */
@Service
@RequiredArgsConstructor
public class AiArtifactService {

    private final SysAiArtifactMapper artifactMapper;

    private final SysFileMapper fileMapper;

    private final SysPredLogMapper predLogMapper;

    private final SysEvalLogMapper evalLogMapper;

    private final StorageServiceFactory storageServiceFactory;

    private final AiConversationService conversationService;

    public IPage<AiArtifactVO> listByConversation(Long convId, Long userId, int pageNum, int pageSize) {
        conversationService.getOwned(convId, userId);
        Page<SysAiArtifact> page = new Page<>(pageNum, pageSize);
        IPage<SysAiArtifact> artifactPage = artifactMapper.selectPage(page, newestFirst()
                .eq(SysAiArtifact::getConversationId, convId));
        return pageOf(page, artifactPage.getRecords().stream().map(this::toVO).toList(), artifactPage.getTotal());
    }

    public List<AiArtifactVO> listByMessage(Long msgId, Long userId) {
        if (conversationService.getOwnedMessage(msgId, userId) == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        return artifactMapper.selectList(newestFirst().eq(SysAiArtifact::getMessageId, msgId))
                .stream().map(this::toVO).toList();
    }

    /**
     * 按业务引用反查产物（仅返回当前用户所属会话的产物）
     */
    public List<AiArtifactVO> listByRef(String refType, Long refId, Long userId) {
        List<SysAiArtifact> artifacts = artifactMapper.selectList(newestFirst()
                .eq(SysAiArtifact::getRefType, refType)
                .eq(SysAiArtifact::getRefId, refId));
        if (artifacts.isEmpty()) {
            return List.of();
        }
        Set<Long> convIds = artifacts.stream().map(SysAiArtifact::getConversationId).collect(Collectors.toSet());
        Set<Long> ownedConvIds = conversationService.listOwnedIds(convIds, userId);
        return artifacts.stream()
                .filter(a -> ownedConvIds.contains(a.getConversationId()))
                .map(this::toVO)
                .toList();
    }

    /**
     * 产物详情：记录 + 运行时图片 URL（URL 不落库，按需经 ref 链路拼接）
     */
    public Map<String, Object> getDetail(Long artifactId, Long userId) {
        SysAiArtifact artifact = artifactMapper.selectById(artifactId);
        if (artifact == null || Integer.valueOf(1).equals(artifact.getIsInvalid())) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "产物不存在或已失效");
        }
        conversationService.getOwned(artifact.getConversationId(), userId);
        Map<String, Object> detail = new LinkedHashMap<>();
        detail.put("artifact", toVO(artifact));
        detail.put("imageUrl", resolveImageUrl(artifact));
        return detail;
    }

    private String resolveImageUrl(SysAiArtifact artifact) {
        Long fileId = null;
        if ("sys_file".equals(artifact.getRefType())) {
            fileId = artifact.getRefId();
        } else if ("sys_pred_log".equals(artifact.getRefType()) && artifact.getRefId() != null) {
            SysPredLog pred = predLogMapper.selectById(artifact.getRefId());
            fileId = pred == null ? null : pred.getPredFileId();
        } else if ("sys_eval_log".equals(artifact.getRefType()) && artifact.getRefId() != null) {
            SysEvalLog evalLog = evalLogMapper.selectById(artifact.getRefId());
            fileId = evalLog == null ? null : evalLog.getPredFileId();
        }
        if (fileId == null) {
            return null;
        }
        SysFile file = fileMapper.selectById(fileId);
        if (file == null) {
            return null;
        }
        return storageServiceFactory.get(file.getStorage()).getUrl(file.getObjectName());
    }

    private LambdaQueryWrapper<SysAiArtifact> newestFirst() {
        return new LambdaQueryWrapper<SysAiArtifact>()
                .orderByDesc(SysAiArtifact::getCreateTime)
                .orderByDesc(SysAiArtifact::getId);
    }

    private AiArtifactVO toVO(SysAiArtifact artifact) {
        AiArtifactVO vo = new AiArtifactVO();
        vo.setId(artifact.getId());
        vo.setConversationId(artifact.getConversationId());
        vo.setMessageId(artifact.getMessageId());
        vo.setType(artifact.getType());
        vo.setRefType(artifact.getRefType());
        vo.setRefId(artifact.getRefId());
        vo.setSummary(artifact.getSummary());
        vo.setIsInvalid(artifact.getIsInvalid());
        vo.setCreateTime(artifact.getCreateTime());
        return vo;
    }

    private <T> IPage<T> pageOf(Page<?> page, List<T> records, long total) {
        Page<T> result = new Page<>(page.getCurrent(), page.getSize(), total);
        result.setRecords(new ArrayList<>(records));
        return result;
    }
}
