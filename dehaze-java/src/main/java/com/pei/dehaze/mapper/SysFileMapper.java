package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysFile;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface SysFileMapper extends BaseMapper<SysFile> {

    /**
     * upsert 文件：md5 唯一键冲突时复活软删行，并用本次上传的新文件信息覆盖业务字段。
     * UPDATE 分支通过 LAST_INSERT_ID(id) 拿回原行 id。
     */
    @Insert("INSERT INTO sys_file (md5, type, name, object_name, storage, size, size_bytes, create_by, deleted, update_time) " +
            "VALUES (#{md5}, #{fileType}, #{fileName}, #{objectName}, #{storage}, #{size}, #{sizeBytes}, #{userId}, 0, NOW()) " +
            "ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), type = VALUES(type), name = VALUES(name), object_name = VALUES(object_name), storage = VALUES(storage), size = VALUES(size), size_bytes = VALUES(size_bytes), deleted = 0, update_time = NOW()")
    int upsertByMd5(@Param("md5") String md5,
                    @Param("fileType") String fileType,
                    @Param("fileName") String fileName,
                    @Param("objectName") String objectName,
                    @Param("storage") String storage,
                    @Param("size") String size,
                    @Param("sizeBytes") Long sizeBytes,
                    @Param("userId") Long userId);

}
