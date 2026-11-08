package com.pei.clarity.common.handler;

import com.fasterxml.jackson.core.type.TypeReference;
import com.pei.clarity.model.dto.ai.AiContract;
import java.util.List;

/**
 * {@code sys_ai_message.sub_agent_usage} 列：契约 {@code SubAgentUsage[]}。
 *
 * @author clarity
 */
public class AiSubAgentUsageListTypeHandler extends AbstractAiJsonTypeHandler<List<AiContract.SubAgentUsage>> {

    public AiSubAgentUsageListTypeHandler() {
        super(new TypeReference<List<AiContract.SubAgentUsage>>() {});
    }
}
