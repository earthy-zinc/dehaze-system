package com.pei.dehaze.model.bo;

import lombok.Data;

import java.util.List;

@Data
public class PairedImage {
    private List<String> hazePath;
    private String cleanPath;
}
