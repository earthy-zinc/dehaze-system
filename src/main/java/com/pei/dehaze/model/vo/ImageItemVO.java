package com.pei.dehaze.model.vo;

import lombok.Data;

import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:46:20
 */
@Data
public class ImageItemVO {
    private Long id;

    private List<ImageUrlVO> imgUrl;
}


