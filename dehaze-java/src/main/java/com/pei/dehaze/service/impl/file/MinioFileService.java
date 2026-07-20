package com.pei.dehaze.service.impl.file;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.service.FileService;
import io.minio.*;
import io.minio.errors.*;
import io.minio.http.Method;
import jakarta.annotation.PostConstruct;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.io.*;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;

/**
 * MinIO 文件上传服务类
 *
 * @author earthyzinc
 * @since 2023/6/2
 */
@Component
@ConditionalOnProperty(value = "file.type", havingValue = "minio")
@ConfigurationProperties(prefix = "file.minio")
@RequiredArgsConstructor
@Data
@Slf4j
public class MinioFileService implements FileService {

    /**
     * 服务Endpoint
     */
    private String endpoint;
    /**
     * 访问凭据
     */
    private String accessKey;
    /**
     * 凭据密钥
     */
    private String secretKey;
    /**
     * 存储桶名称
     */
    private String bucketName;

    private MinioClient minioClient;

    @Value("${file.baseUrl}")
    private String baseUrl;

    /**
     * 标记桶是否已初始化（用于延迟创建）
     */
    private volatile boolean bucketInitialized = false;

    // 依赖注入完成之后执行初始化（带重试，MinIO 不可达时不阻塞应用启动）
    @PostConstruct
    public void init() {
        minioClient = MinioClient.builder()
                .endpoint(endpoint)
                .credentials(accessKey, secretKey)
                .build();

        // 重试3次，每次间隔2秒；全部失败时仅记录警告，不阻塞启动
        for (int attempt = 1; attempt <= 3; attempt++) {
            try {
                createBucketIfAbsent(bucketName);
                bucketInitialized = true;
                return;
            } catch (Exception e) {
                log.warn("MinIO 初始化失败 (attempt={}/3): {}", attempt, e.getMessage());
                if (attempt < 3) {
                    try {
                        Thread.sleep(2000L);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }
        log.warn("MinIO 存储桶初始化失败，将在首次文件操作时重试。endpoint={}", endpoint);
    }

    /**
     * 确保桶已初始化（延迟创建，用于启动时 MinIO 不可达的场景）
     */
    private void ensureBucketInitialized() {
        if (!bucketInitialized) {
            synchronized (this) {
                if (!bucketInitialized) {
                    createBucketIfAbsent(bucketName);
                    bucketInitialized = true;
                }
            }
        }
    }

    @Override
    public FileBO uploadFile(FileBO fileBO) {
        ensureBucketInitialized();
        String objectName = fileBO.getObjectName();
        String mimeType = FileUtil.getMimeType(fileBO.getName());
        Assert.notBlank(objectName);
        Assert.notBlank(mimeType);

        File file = fileBO.getFile();
        try (FileInputStream stream = new FileInputStream(file)){
            PutObjectArgs putObjectArgs = PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .contentType(mimeType)
                    .stream(stream, stream.available(), -1)
                    .build();
            minioClient.putObject(putObjectArgs);
            String url = getUrl(objectName);
            fileBO.setUrl(url);
            return fileBO;
        } catch (Exception e) {
            throw new BusinessException("无法保存文件", e);
        }
    }

    @Override
    public String uploadFile(String objectName, InputStream inputStream, long fileSize, String contentType) {
        ensureBucketInitialized();
        Assert.notBlank(objectName, "objectName不能为空");
        Assert.notNull(inputStream, "inputStream不能为空");

        try {
            PutObjectArgs putObjectArgs = PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .contentType(contentType != null ? contentType : "application/octet-stream")
                    .stream(inputStream, fileSize, -1)
                    .build();
            minioClient.putObject(putObjectArgs);
            return getUrl(objectName);
        } catch (Exception e) {
            throw new BusinessException("无法保存文件: " + e.getMessage(), e);
        }
    }

    private String getUrl(String objectName) throws ErrorResponseException, InsufficientDataException, InternalException, InvalidKeyException, InvalidResponseException, IOException, NoSuchAlgorithmException, XmlParserException, ServerException {
        // 返回文件路径
        String fileUrl;
        if (CharSequenceUtil.isBlank(baseUrl)) { // 未配置自定义域名
            GetPresignedObjectUrlArgs getPresignedObjectUrlArgs = GetPresignedObjectUrlArgs.builder()
                    .bucket(bucketName).object(objectName)
                    .method(Method.GET)
                    .build();

            fileUrl = minioClient.getPresignedObjectUrl(getPresignedObjectUrlArgs);
            fileUrl = fileUrl.substring(0, fileUrl.indexOf("?"));
        } else { // 配置自定义文件路径域名
            fileUrl = baseUrl + "/" + objectName;
        }
        return fileUrl;
    }

    /**
     * 删除文件
     *
     * @param objectName 文件 objectName
     * @return 是否删除成功
     */
    @Override
    public boolean deleteFile(String objectName) {
        Assert.notBlank(objectName, "删除文件objectName不能为空");
        try {
            RemoveObjectArgs removeObjectArgs = RemoveObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .build();
            minioClient.removeObject(removeObjectArgs);
            return true;
        } catch (ErrorResponseException | InsufficientDataException | InternalException | InvalidKeyException |
                 InvalidResponseException | IOException | NoSuchAlgorithmException | ServerException |
                 XmlParserException e) {
            throw new BusinessException("删除文件失败", e);
        }
    }

    @Override
    public InputStream downLoadFile(String objectName) {
        GetObjectArgs getObjectArgs = GetObjectArgs.builder()
                .bucket(bucketName)
                .object(objectName)
                .build();
        try {
            // 直接返回响应流，避免将整个文件读入内存导致OOM
            // GetObjectResponse 继承自 FilterInputStream，调用方负责关闭返回的 InputStream
            return minioClient.getObject(getObjectArgs);
        } catch (ErrorResponseException | InsufficientDataException | InternalException | InvalidKeyException |
                 InvalidResponseException | IOException | NoSuchAlgorithmException | ServerException |
                 XmlParserException e) {
            throw new BusinessException("下载文件失败: " + e.getMessage(), e);
        }
    }

    /**
     * 创建存储桶(存储桶不存在)
     *
     * @param bucketName 存储桶名称
     */
    private void createBucketIfAbsent(String bucketName) {
        try {
            BucketExistsArgs bucketExistsArgs = BucketExistsArgs.builder().bucket(bucketName).build();
            if (!minioClient.bucketExists(bucketExistsArgs)) {
                MakeBucketArgs makeBucketArgs = MakeBucketArgs.builder().bucket(bucketName).build();

                minioClient.makeBucket(makeBucketArgs);

                // 设置存储桶访问权限为PUBLIC， 如果不配置，则新建的存储桶默认是PRIVATE，则存储桶文件会拒绝访问 Access Denied
                SetBucketPolicyArgs setBucketPolicyArgs = SetBucketPolicyArgs
                        .builder()
                        .bucket(bucketName)
                        .config(publicBucketPolicy(bucketName))
                        .build();
                minioClient.setBucketPolicy(setBucketPolicyArgs);
            }
        } catch (ErrorResponseException | InsufficientDataException | InternalException | InvalidKeyException |
                 InvalidResponseException | IOException | NoSuchAlgorithmException | ServerException |
                 XmlParserException e) {
            throw new BusinessException("初始化MinIO存储桶失败: " + e.getMessage(), e);
        }
    }

    /**
     * PUBLIC桶策略（只读）
     * 仅允许匿名 GetObject（用于文件预览/下载），不允许匿名写入和删除
     * 写入/删除操作通过服务端 MinioClient（携带 AccessKey/SecretKey）执行
     *
     * @param bucketName 存储桶名称
     */
    private static String publicBucketPolicy(String bucketName) {
        /*
         * AWS的S3存储桶策略
         * Principal: 生效用户对象
         * Resource:  指定存储桶
         * Action: 操作行为（仅允许只读访问）
         */
        return "{\"Version\":\"2012-10-17\","
                + "\"Statement\":[{\"Effect\":\"Allow\","
                + "\"Principal\":{\"AWS\":[\"*\"]},"
                + "\"Action\":[\"s3:GetBucketLocation\",\"s3:ListBucket\"],"
                + "\"Resource\":[\"arn:aws:s3:::" + bucketName + "\"]},"
                + "{\"Effect\":\"Allow\"," + "\"Principal\":{\"AWS\":[\"*\"]},"
                + "\"Action\":[\"s3:GetObject\"],"
                + "\"Resource\":[\"arn:aws:s3:::" + bucketName + "/*\"]}]}";
    }
}
