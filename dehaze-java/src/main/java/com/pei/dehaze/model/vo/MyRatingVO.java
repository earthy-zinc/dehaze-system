package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Schema(description = "我的评价列表VO")
public class MyRatingVO {

    @Schema(description = "评价ID")
    private Long id;

    @Schema(description = "处理记录ID")
    private Long predLogId;

    @Schema(description = "算法名称")
    private String algorithmName;

    @Schema(description = "评分(1-5)")
    private Integer rating;

    @Schema(description = "评价文字")
    private String comment;

    @Schema(description = "评价标签")
    private List<String> tags;

    @Schema(description = "截图URL")
    private List<String> imageUrls;

    @Schema(description = "是否匿名(0:否;1:是)")
    private Integer isAnonymous;

    @Schema(description = "管理员回复")
    private String adminReply;

    @Schema(description = "回复时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime replyTime;

    @Schema(description = "评价时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
