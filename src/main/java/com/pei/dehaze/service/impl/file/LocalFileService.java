package com.pei.dehaze.service.impl.file;

import com.pei.dehaze.model.dto.FileInfo;
import com.pei.dehaze.service.FileService;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

/**
 * @author earthy-zinc
 * @since 2024-06-08 19:24:03
 */
@Component
@ConditionalOnProperty(value = "file.type", havingValue = "local")
@ConfigurationProperties(prefix = "file.local")
@RequiredArgsConstructor
@Data
public class LocalFileService implements FileService {

    public boolean uploadCheck(String md5) {
        return true;
    }

    /**
     * 上传文件
     *
     * @param file 表单文件对象
     * @return 文件信息
     */
    @Override
    public FileInfo uploadFile(MultipartFile file) {
        return null;
    }

    /**
     * 删除文件
     *
     * @param filePath 文件完整URL
     * @return 删除结果
     */
    @Override
    public boolean deleteFile(String filePath) {
        return false;
    }
}
