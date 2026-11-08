package com.pei.clarity.common.handler;

import com.fasterxml.jackson.core.type.TypeReference;
import com.pei.clarity.model.dto.ai.AiContract;
import java.util.List;

/**
 * {@code sys_ai_message.blocks} 列：契约 {@code Block[]}。
 *
 * @author clarity
 */
public class AiBlockListTypeHandler extends AbstractAiJsonTypeHandler<List<AiContract.Block>> {

    public AiBlockListTypeHandler() {
        super(new TypeReference<List<AiContract.Block>>() {});
    }
}
