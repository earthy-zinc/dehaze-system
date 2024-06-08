package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * @author earthy-zinc
 * @since 2024-06-08 19:19:08
 */
@Data
public class AlgorithmQuery {

    @Schema(description = "关键字")
    private String keywords;
}
