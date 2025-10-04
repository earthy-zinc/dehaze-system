package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:46:20
 */
@Data
@Document(collection = "image_item")
public class ImageItemVO {

    @Id
    @Schema(description = "DatasetItemId")
    private Long id;

    @Indexed
    private Long datasetId;

    private List<ImageUrlVO> imgUrl;
}


