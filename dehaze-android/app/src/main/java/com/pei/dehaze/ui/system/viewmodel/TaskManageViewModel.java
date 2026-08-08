package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.TaskAPI;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskVO;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.List;

public class TaskManageViewModel extends BaseViewModel {

    private final MutableLiveData<List<TaskVO>> taskList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);

    private int pageNum = 1;
    private final int pageSize = 10;
    private String keywords = "";

    public void loadTasks() {
        TaskQuery query = new TaskQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        TaskAPI.getTaskPage(query, RepositoryAdapters.wrap(withLoading(data -> {
            taskList.postValue(data.getList());
            total.postValue(data.getTotal());
        })));
    }

    public void cancelTask(String taskId) {
        TaskAPI.cancelTask(taskId, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("任务已取消");
            loadTasks();
        })));
    }

    public void setKeywords(String keywords) {
        this.keywords = keywords != null ? keywords : "";
        this.pageNum = 1;
    }

    public void resetQuery() {
        this.keywords = "";
        this.pageNum = 1;
    }

    public void nextPage() {
        long t = total.getValue() != null ? total.getValue() : 0L;
        if (pageNum < (int) Math.ceil(t * 1.0 / pageSize)) {
            pageNum++;
            loadTasks();
        }
    }

    public void prevPage() {
        if (pageNum > 1) {
            pageNum--;
            loadTasks();
        }
    }

    public int getPageNum() { return pageNum; }
    public int getPageSize() { return pageSize; }

    public LiveData<List<TaskVO>> getTaskList() { return taskList; }
    public LiveData<Long> getTotal() { return total; }
}
