package com.pei.dehaze.service.impl.file;

import cn.hutool.core.date.DateUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.core.util.IdUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.model.dto.ImageFileInfo;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.form.ImageForm;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysFileService;
import io.minio.*;
import io.minio.errors.*;
import io.minio.http.Method;
import jakarta.annotation.PostConstruct;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;

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
    /**
     * 自定义域名
     */
    private String customDomain;

    private MinioClient minioClient;

    private SysFileService sysFileService;

    // 依赖注入完成之后执行初始化
    @PostConstruct
    public void init() {
        minioClient = MinioClient.builder()
                .endpoint(endpoint)
                .credentials(accessKey, secretKey)
                .build();
    }


    /**
     * 文件上传检查
     *
     * @param md5 文件md5
     * @return true 表示文件已存在
     */
    @Override
    public boolean uploadCheck(String md5) {
        return sysFileService.lambdaQuery().eq(SysFile::getMd5, md5).getEntity() != null;
    }

    /**
     * 上传文件
     *
     * @param file 表单文件对象
     * @return 文件信息
     */
    @Override
    public SysFile uploadFile(MultipartFile file) {
        // 生成文件名(日期文件夹)
        String fileName = file.getOriginalFilename();
        String fileSize = FileUtil.readableFileSize(file.getSize());
        String suffix = FileUtil.getSuffix(fileName);
        String uuid = IdUtil.simpleUUID();
        String objectName = DateUtil.format(LocalDateTime.now(), "yyyyMMdd") + File.separator + uuid + "." + suffix;
        //  try-with-resource 语法糖自动释放流
        try (InputStream inputStream = file.getInputStream()) {
            // 检查文件md5
            String md5 = FileUploadUtils.getMd5(inputStream);
            // 从SysFile中查询是否存在md5相同的数据
            SysFile foundFile = sysFileService.lambdaQuery().eq(SysFile::getMd5, md5).getEntity();

            if (foundFile != null) return foundFile;

            // 文件上传
            PutObjectArgs putObjectArgs = PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .contentType(file.getContentType())
                    .stream(inputStream, inputStream.available(), -1)
                    .build();
            minioClient.putObject(putObjectArgs);

            // 返回文件路径
            String fileUrl;
            if (CharSequenceUtil.isBlank(customDomain)) { // 未配置自定义域名
                GetPresignedObjectUrlArgs getPresignedObjectUrlArgs = GetPresignedObjectUrlArgs.builder()
                        .bucket(bucketName).object(objectName)
                        .method(Method.GET)
                        .build();

                fileUrl = minioClient.getPresignedObjectUrl(getPresignedObjectUrlArgs);
                fileUrl = fileUrl.substring(0, fileUrl.indexOf("?"));
            } else { // 配置自定义文件路径域名
                fileUrl = customDomain + '/' + bucketName + "/" + fileName;
            }

            // 保存文件信息到数据库
            SysFile sysFile = SysFile.builder()
                    .name(fileName)
                    .objectName(objectName)
                    .size(fileSize)
                    .type(suffix)
                    .url(fileUrl)
                    .md5(md5)
                    .path("")
                    .build();
            sysFileService.save(sysFile);

            return sysFile;
        } catch (Exception e) {
            throw new BusinessException("文件上传失败");
        }
    }

    @Override
    public SysFile getWpxFile(SysFile oldFile, Long modelId) {
        return null;
    }

    @Override
    public ImageFileInfo uploadImage(MultipartFile file, ImageForm imageForm) {
        return null;
    }


    /**
     * 删除文件
     *
     * @param filePath 文件路径（文件URL）
     * @return 是否删除成功
     */
    @Override
    public boolean deleteFile(String filePath) {
        Assert.notBlank(filePath, "删除文件路径不能为空");
        try {
            // 获取数据库 sysFile 中文件信息
            SysFile sysFile = sysFileService.lambdaQuery().eq(SysFile::getUrl, filePath).getEntity();

            // 删除文件
            String objectName = sysFile.getObjectName();
            RemoveObjectArgs removeObjectArgs = RemoveObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .build();
            minioClient.removeObject(removeObjectArgs);

            // 删除数据库中对应的数据
            sysFileService.removeById(sysFile);
            return true;
        } catch (ErrorResponseException | InsufficientDataException | InternalException | InvalidKeyException |
                 InvalidResponseException | IOException | NoSuchAlgorithmException | ServerException |
                 XmlParserException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 删除图片时，会将连带的数据项下对应的所有图片都删除
     *
     * @param filePath 当前图片在数据库 SysFile 数据表中的 URL
     */
    @Override
    public boolean deleteImage(String filePath) {
        return false;
    }

    /**
     * 创建存储桶(存储桶不存在)
     *
     * @param bucketName 存储桶名称
     */
    @SneakyThrows
    private void createBucketIfAbsent(String bucketName) {
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
    }

    /**
     * PUBLIC桶策略
     * 如果不配置，则新建的存储桶默认是PRIVATE，则存储桶文件会拒绝访问 Access Denied
     *
     * @param bucketName 存储桶名称
     */
    private static String publicBucketPolicy(String bucketName) {
        /*
         * AWS的S3存储桶策略
         * Principal: 生效用户对象
         * Resource:  指定存储桶
         * Action: 操作行为
         */
        return "{\"Version\":\"2012-10-17\","
                + "\"Statement\":[{\"Effect\":\"Allow\","
                + "\"Principal\":{\"AWS\":[\"*\"]},"
                + "\"Action\":[\"s3:ListBucketMultipartUploads\",\"s3:GetBucketLocation\",\"s3:ListBucket\"],"
                + "\"Resource\":[\"arn:aws:s3:::" + bucketName + "\"]},"
                + "{\"Effect\":\"Allow\"," + "\"Principal\":{\"AWS\":[\"*\"]},"
                + "\"Action\":[\"s3:ListMultipartUploadParts\",\"s3:PutObject\",\"s3:AbortMultipartUpload\",\"s3:DeleteObject\",\"s3:GetObject\"],"
                + "\"Resource\":[\"arn:aws:s3:::" + bucketName + "/*\"]}]}";
    }
}
