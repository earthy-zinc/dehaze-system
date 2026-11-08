package com.pei.clarity.common.handler;

import com.fasterxml.jackson.core.type.TypeReference;
import com.pei.clarity.model.dto.ai.AiContract;

/**
 * {@code sys_ai_message.pending_interrupt} 列：契约 {@code Interrupt}。
 *
 * @author clarity
 */
public class AiInterruptTypeHandler extends AbstractAiJsonTypeHandler<AiContract.Interrupt> {

    public AiInterruptTypeHandler() {
        super(new TypeReference<AiContract.Interrupt>() {});
    }
}
