package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:54:55
 */
@Data
public class DatasetQuery {

    @Schema(description = "关键字")
    private String keywords;

}
