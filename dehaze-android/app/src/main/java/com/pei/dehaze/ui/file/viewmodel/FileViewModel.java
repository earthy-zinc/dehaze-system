package com.pei.dehaze.ui.file.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.FileRepository;
import com.pei.dehaze.sdk.model.file.FileInfo;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 文件管理 ViewModel
 */
public class FileViewModel extends ViewModel {

    private final FileRepository fileRepository;

    private final MutableLiveData<List<FileInfo>> fileList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();
    private final MutableLiveData<FileInfo> fileDetail = new MutableLiveData<>();

    private int pageNum = 1;
    private int pageSize = 10;
    private String keywords = "";
    private long total = 0;

    public FileViewModel() {
        fileRepository = new FileRepository();
    }

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
        loading.setValue(true);
        fileRepository.getFiles(pageNum, pageSize, keywords, new FileRepository.FileListCallback() {
            @Override
            public void onSuccess(List<FileInfo> files, long total) {
                FileViewModel.this.total = total;
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
                loading.postValue(false);
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("[" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 上传文件
     */
    public void uploadFile(File file) {
        loading.setValue(true);
        fileRepository.uploadFile(file, new FileRepository.FileCallback() {
            @Override
            public void onSuccess(FileInfo file) {
                operationResult.postValue("上传成功: " + file.getName());
                loading.postValue(false);
                loadFiles();
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("上传失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 下载文件
     */
    public void downloadFile(FileInfo file) {
        loading.setValue(true);
        fileRepository.downloadFile(file.getObjectName(), file.getName(), new FileRepository.ActionCallback() {
            @Override
            public void onSuccess() {
                operationResult.postValue("下载成功，已保存到下载目录");
                loading.postValue(false);
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("下载失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 删除文件
     */
    public void deleteFile(long fileId) {
        loading.setValue(true);
        fileRepository.deleteFile(fileId, new FileRepository.ActionCallback() {
            @Override
            public void onSuccess() {
                operationResult.postValue("删除成功");
                loading.postValue(false);
                loadFiles();
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("删除失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 查看文件详情
     */
    public void getFileDetail(long fileId) {
        loading.setValue(true);
        fileRepository.getFileDetail(fileId, new FileRepository.FileCallback() {
            @Override
            public void onSuccess(FileInfo file) {
                fileDetail.postValue(file);
                loading.postValue(false);
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("查询详情失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    public LiveData<List<FileInfo>> getFileList() {
        return fileList;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<String> getOperationResult() {
        return operationResult;
    }

    public LiveData<FileInfo> getFileDetail() {
        return fileDetail;
    }

    public long getTotal() {
        return total;
    }
}
