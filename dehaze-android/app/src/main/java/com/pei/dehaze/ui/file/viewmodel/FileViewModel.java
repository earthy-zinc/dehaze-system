package com.pei.dehaze.ui.file.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.FileRepository;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.file.FileInfo;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 文件管理 ViewModel
 */
public class FileViewModel extends BaseViewModel {

    private final FileRepository fileRepository = new FileRepository();

    private final MutableLiveData<List<FileInfo>> fileList = new MutableLiveData<>();
    private final MutableLiveData<FileInfo> fileDetail = new MutableLiveData<>();

    private int pageNum = 1;
    private int pageSize = 10;
    private String keywords = "";
    private long total = 0;

    /**
     * 加载文件列表（首页）
     */
    public void loadFiles() {
        pageNum = 1;
        fetchFiles();
    }

    /**
     * 搜索文件
     */
    public void searchFiles(String keywords) {
        this.keywords = keywords;
        pageNum = 1;
        fetchFiles();
    }

    /**
     * 加载下一页
     */
    public void loadMore() {
        if (fileList.getValue() == null || fileList.getValue().size() >= total) {
            return;
        }
        pageNum++;
        fetchFiles();
    }

    private void fetchFiles() {
        FileAPI.getFilePage(pageNum, pageSize, keywords, RepositoryAdapters.wrap(withLoading(data -> {
            List<FileInfo> files = data.getList();
            FileViewModel.this.total = data.getTotal();
            if (pageNum == 1) {
                fileList.postValue(files);
            } else {
                List<FileInfo> current = fileList.getValue();
                if (current == null) {
                    current = new ArrayList<>();
                }
                current.addAll(files);
                fileList.postValue(current);
            }
        })));
    }

    /**
     * 上传文件
     */
    public void uploadFile(File file) {
        FileAPI.upload(file, RepositoryAdapters.wrap(withLoading(f -> {
            operationResult.postValue("上传成功: " + f.getName());
            loadFiles();
        }, msg -> error.postValue("上传失败: " + msg))));
    }

    /**
     * 下载文件
     */
    public void downloadFile(FileInfo file) {
        fileRepository.downloadFile(file.getObjectName(), file.getName(), withLoading(v ->
                operationResult.postValue("下载成功，已保存到下载目录"),
                msg -> error.postValue("下载失败: " + msg)));
    }

    /**
     * 删除文件
     */
    public void deleteFile(long fileId) {
        FileAPI.delete(fileId, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除成功");
            loadFiles();
        }, msg -> error.postValue("删除失败: " + msg))));
    }

    /**
     * 查看文件详情
     */
    public void getFileDetail(long fileId) {
        FileAPI.getFileDetail(fileId, RepositoryAdapters.wrap(withLoading(fileDetail::postValue,
                msg -> error.postValue("查询详情失败: " + msg))));
    }

    public LiveData<List<FileInfo>> getFileList() {
        return fileList;
    }

    public LiveData<FileInfo> getFileDetail() {
        return fileDetail;
    }

    public long getTotal() {
        return total;
    }
}
