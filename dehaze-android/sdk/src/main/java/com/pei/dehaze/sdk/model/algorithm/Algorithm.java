package com.pei.dehaze.sdk.model.algorithm;

import lombok.Data;
import java.util.List;

/**
 * 算法模型类
 */
@Data
public class Algorithm {
    private Long id;
    private Long parentId;
    private String name;
    private String type;
    private String description;
    private String img;
    private String path;
    private String importPath;
    private String params;
    private String flops;
    private AlgorithmStatus status;
    private String size;
    private List<Algorithm> children;
}