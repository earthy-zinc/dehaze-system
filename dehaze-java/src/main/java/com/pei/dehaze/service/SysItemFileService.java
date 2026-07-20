package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.BatchDeleteForm;
import com.pei.dehaze.model.form.ItemFileUpdateForm;
import com.pei.dehaze.model.vo.BatchDeleteResultVO;
import com.pei.dehaze.model.vo.ImageUrlVO;

import java.util.List;
import java.util.Map;

public interface SysItemFileService extends IService<SysItemFile> {

    /**
     * 保存数据项图片
     *
     * @param itemId 所属数据项ID
     * @param itemBO 图片业务对象
     * @return 图片信息VO
     */
    ImageUrlVO saveItemFile(Long itemId, ItemFileBO itemBO);

    /**
     * 短事务写入数据项文件关联记录（将 DB 写入与 MinIO 上传分离）
     *
     * @param itemId           所属数据项ID
     * @param itemBO           图片业务对象
     * @param sysFile          源文件实体
     * @param thumbnailSysFile 缩略图文件实体
     * @return 数据项文件实体
     */
    SysItemFile saveItemFileRecord(Long itemId, ItemFileBO itemBO, SysFile sysFile, SysFile thumbnailSysFile);

    /**
     * 获取指定数据项的图片列表
     *
     * @param itemId 数据项ID
     * @return 图片URL列表
     */
    List<ImageUrlVO> getImageUrlVOs(Long itemId);

    /**
     * 删除图片
     *
     * @param id 图片ID
     * @return 是否删除成功
     */
    boolean deleteFile(Long id);

    /**
     * 短事务删除数据项文件 DB 记录（MinIO 删除在事务外完成后调用）
     *
     * @param id 图片ID
     * @return 是否删除成功
     */
    boolean deleteFileRecord(Long id);

    /**
     * 批量删除图片
     *
     * @param ids 图片ID列表
     * @return 批量删除结果
     */
    BatchDeleteResultVO batchDelete(List<Long> ids);

    /**
     * 获取图片详情（包含配对图片和数据项信息）
     *
     * @param id 图片ID
     * @return 图片详情VO（包含配对图片列表和数据项信息）
     */
    ImageUrlVO getImageById(Long id);

    /**
     * 修改图片信息（包含标注信息）
     * 合并了原来的"修改图片信息"和"图片标注"功能
     *
     * @param id 图片ID
     * @param form 图片更新表单
     * @return 更新结果
     */
    boolean updateItemFileInfo(Long id, ItemFileUpdateForm form);

    /**
     * 增加图片使用次数
     *
     * @param id 图片ID
     */
    void incrementUsageCount(Long id);

    /**
     * 批量预加载文件信息，构建 fileId -> SysFile 的Map
     * 用于避免转换VO时的N+1查询
     *
     * @param itemFiles 数据项文件列表
     * @return 文件Map（fileId -> SysFile）
     */
    Map<Long, SysFile> buildFileMap(List<SysItemFile> itemFiles);

    /**
     * 将SysItemFile实体转换为ImageUrlVO
     * 使用预加载的文件Map避免N+1查询
     *
     * @param itemFile 数据项文件实体
     * @param fileMap  预加载的文件Map（fileId -> SysFile），由调用方通过 {@link #buildFileMap} 批量构建
     * @return 图片URL信息VO
     */
    ImageUrlVO convertToImageUrlVO(SysItemFile itemFile, Map<Long, SysFile> fileMap);
}
