package com.pei.dehaze.sdk.model.feedback;

import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class RatingPageVO extends MyRatingVO {
    private Long userId;
    private String username;
    private String userAvatar;
    private Integer isHidden;
}
