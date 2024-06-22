package com.pei.dehaze.model.dto;

import lombok.Data;

import java.util.List;

@Data
public class ImageFileInfo {
    private Long datasetId;
    private String type;
    private String path;
    private String relativePath;
    private List<String> imageNames;
}
