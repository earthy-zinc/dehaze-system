package com.pei.dehaze.service;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiAgentMapper;
import com.pei.dehaze.mapper.SysAiAgentVersionMapper;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysAiUserPreferenceMapper;
import com.pei.dehaze.mapper.SysDictMapper;
import com.pei.dehaze.model.entity.SysAiAgent;
import com.pei.dehaze.model.entity.SysAiAgentVersion;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.entity.SysAiUserPreference;
import com.pei.dehaze.model.entity.SysDict;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * 默认对话模型解析链与用户偏好校验。
 *
 * <p>解析链（三端全等）：请求/会话显式指定的 model ＞ 用户偏好 default_model ＞ Agent 配置模型
 * （会话锚定 Agent 的已发布版本快照 model_id，无已发布版本回退主表）＞ 平台默认模型
 * （{@code sys_dict} 的 {@code ai_model.default_model} 启用字典值）＞ 未配置报 A0500，
 * 绝不回落代码常量。用户偏好/Agent 模型/平台默认都只在"未指定模型"时参与解析。
 *
 * <p>用户偏好的写入校验（{@link #assertSelectable}）与解析链共用同一套可用性口径
 * （存在 + status=1 + model_type='chat'），等级校验复用 {@link AiUserLevelResolver}，
 * 与启用模型列表过滤同源。
 *
 * @author dehaze
 */
@Component
@RequiredArgsConstructor
public class AiDefaultModelResolver {

    private static final String PLATFORM_DEFAULT_DICT_TYPE = "ai_model";
    private static final String PLATFORM_DEFAULT_DICT_NAME = "default_model";

    private final SysAiUserPreferenceMapper preferenceMapper;

    private final SysAiModelMapper modelMapper;

    private final SysDictMapper dictMapper;

    private final SysAiAgentMapper agentMapper;

    private final SysAiAgentVersionMapper agentVersionMapper;

    private final AiUserLevelResolver userLevelResolver;

    /**
     * 某模型类型的平台默认模型字典键：chat 用 {@code default_model}，其余
     * {@code default_model.{modelType}}（与 config/sql/data/sys_dict.sql 键约定一致，
     * 亦与 python {@code model_defaults.default_model_dict_key} 同源）。
     */
    public static String dictKeyFor(String modelType) {
        return "chat".equals(modelType) ? PLATFORM_DEFAULT_DICT_NAME : PLATFORM_DEFAULT_DICT_NAME + "." + modelType;
    }

    /**
     * 某模型类型的平台默认模型标识（启用字典值）；未配置/已停用/空值返回 null 而不报错。
     *
     * <p>仅供"读取默认模型标记"等展示场景使用（如启用模型列表的 isDefault）：真正需要
     * 模型的解析链路仍走 {@link #resolveForUser}，未配置即报 A0500，不回落代码常量。
     */
    public String resolvePlatformDefaultModelId(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return null;
        }
        SysDict dict = dictMapper.selectOne(new LambdaQueryWrapper<SysDict>()
                .eq(SysDict::getTypeCode, PLATFORM_DEFAULT_DICT_TYPE)
                .eq(SysDict::getName, dictKeyFor(modelType))
                .eq(SysDict::getStatus, 1)
                .orderByAsc(SysDict::getId)
                .last("LIMIT 1"));
        if (dict == null || CharSequenceUtil.isBlank(dict.getValue())) {
            return null;
        }
        return dict.getValue();
    }

    /** 取该用户的默认对话模型；偏好模型不可用则静默跳过，平台默认未配置抛 A0500 */
    public String resolveForUser(Long userId, String agentCode) {
        SysAiUserPreference preference = preferenceMapper.selectOne(
                new LambdaQueryWrapper<SysAiUserPreference>().eq(SysAiUserPreference::getUserId, userId));
        if (preference != null && CharSequenceUtil.isNotBlank(preference.getDefaultModel())) {
            SysAiModel preferred = findEnabledChatModel(preference.getDefaultModel());
            if (preferred != null) {
                return preferred.getModelId();
            }
        }
        String agentModel = resolveAgentModel(agentCode);
        if (CharSequenceUtil.isNotBlank(agentModel)) {
            return agentModel;
        }
        String platformDefault = resolvePlatformDefaultModelId("chat");
        if (platformDefault == null) {
            throw new BusinessException(
                    ResultCode.BUSINESS_ERROR, "未配置平台默认模型（sys_dict: ai_model.default_model），请先在 AI 模型管理/字典管理中配置");
        }
        return platformDefault;
    }

    /**
     * Agent 配置模型：已发布版本快照 model_id 优先（运行面口径，版本化），无已发布版本回退主表可编辑态；
     * Agent 不存在/无模型返回 null（跳到下一级）。不做启用性判断——Agent 显式配置的模型不可用时由消息入口
     * 按统一口径报错，静默回落会掩盖管理端的配置错误。
     */
    private String resolveAgentModel(String agentCode) {
        if (CharSequenceUtil.isBlank(agentCode)) {
            return null;
        }
        SysAiAgent agent = agentMapper.selectOne(new LambdaQueryWrapper<SysAiAgent>()
                .eq(SysAiAgent::getAgentCode, agentCode.trim())
                .last("LIMIT 1"));
        if (agent == null) {
            return null;
        }
        SysAiAgentVersion version = agentVersionMapper.selectOne(new LambdaQueryWrapper<SysAiAgentVersion>()
                .eq(SysAiAgentVersion::getAgentId, agent.getId())
                .eq(SysAiAgentVersion::getStatus, 2)
                .orderByDesc(SysAiAgentVersion::getVersionNo)
                .last("LIMIT 1"));
        if (version != null && version.getSnapshot() != null) {
            Object snapshotModel = version.getSnapshot().get("model_id");
            if (snapshotModel != null && CharSequenceUtil.isNotBlank(String.valueOf(snapshotModel))) {
                return String.valueOf(snapshotModel);
            }
        }
        return CharSequenceUtil.isNotBlank(agent.getModelId()) ? agent.getModelId() : null;
    }

    /** 校验模型可作为该用户的偏好默认模型：不存在/未启用/非 chat/超出会员等级一律 A0601 */
    public void assertSelectable(String modelId, Long userId) {
        SysAiModel model = findEnabledChatModel(modelId);
        if (model == null || (model.getVipLevel() != null && model.getVipLevel() > userLevelResolver.levelOf(userId))) {
            throw new BusinessException(ResultCode.AI_MODEL_NOT_AVAILABLE, "模型不存在、未启用或当前等级不可用");
        }
    }

    /** 启用中的 chat 模型；同一 model_id 可配多供应商，取主键最小者 */
    private SysAiModel findEnabledChatModel(String modelId) {
        return modelMapper.selectOne(new LambdaQueryWrapper<SysAiModel>()
                .eq(SysAiModel::getModelId, modelId)
                .eq(SysAiModel::getStatus, 1)
                .eq(SysAiModel::getModelType, "chat")
                .orderByAsc(SysAiModel::getId)
                .last("LIMIT 1"));
    }
}
