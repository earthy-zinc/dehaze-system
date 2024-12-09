package com.pei.dehaze.model.bo;

import lombok.Data;

import java.io.File;

@Data
public class FileBO {
    // 文件输入流
    private File file;
    // 文件名
    private String name;
    // 文件对象名
    private String objectName;
    // 文件扩展名
    private String extension;
    // 文件MD5值
    private String md5;
    // 文件路径
    private String path;
    // 文件大小
    private Long size;

    private String url;
}
