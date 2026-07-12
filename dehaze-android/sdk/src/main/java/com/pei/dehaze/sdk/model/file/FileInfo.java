package com.pei.dehaze.sdk.model.file;

import lombok.Data;

/**
 * 文件信息模型类（对齐后端 SysFile）
 */
@Data
public class FileInfo {
    private Long id;
    private String type;
    private String url;
    private String name;
    private String objectName;
    private String size;
    private String path;
    private String md5;
    private String createTime;
    private String updateTime;
}
