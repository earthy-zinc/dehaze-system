package com.pei.clarity.common.handler;

import com.fasterxml.jackson.core.type.TypeReference;
import com.pei.clarity.model.dto.ai.AiContract;

/**
 * {@code sys_ai_message.plan} 列：契约 {@code Plan}。
 *
 * @author clarity
 */
public class AiPlanTypeHandler extends AbstractAiJsonTypeHandler<AiContract.Plan> {

    public AiPlanTypeHandler() {
        super(new TypeReference<AiContract.Plan>() {});
    }
}
