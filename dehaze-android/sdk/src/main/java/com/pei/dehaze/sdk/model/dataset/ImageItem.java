package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;
import java.util.List;

/**
 * 图片项模型类
 */
@Data
public class ImageItem {
    private int id;
    private List<ImageUrl> imgUrl;
}