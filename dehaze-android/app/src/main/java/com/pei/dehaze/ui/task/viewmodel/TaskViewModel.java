package com.pei.dehaze.ui.task.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.TaskRepository;
import com.pei.dehaze.sdk.api.TaskAPI;
import com.pei.dehaze.ui.common.BaseLoadMoreViewModel;
import com.pei.dehaze.sdk.model.task.TaskCreateForm;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskStatus;
import com.pei.dehaze.sdk.model.task.TaskType;
import com.pei.dehaze.sdk.model.task.TaskVO;

import java.util.List;

/**
 * 任务管理 ViewModel
 */
public class TaskViewModel extends BaseLoadMoreViewModel<TaskVO> {

    private final TaskRepository taskRepository = new TaskRepository();

    private final MutableLiveData<TaskVO> taskDetail = new MutableLiveData<>();
    private final MutableLiveData<TaskVO> createdTask = new MutableLiveData<>();

    private TaskStatus statusFilter;
    private TaskType typeFilter;

    public TaskViewModel() {
        super(10);
    }

    /**
     * 加载任务列表（首页）
     */
    public void loadTasks() {
        reload();
    }

    /**
     * 按状态筛选
     */
    public void filterByStatus(TaskStatus status) {
        this.statusFilter = status;
        reload();
    }

    /**
     * 按类型筛选
     */
    public void filterByType(TaskType taskType) {
        this.typeFilter = taskType;
        reload();
    }

    @Override
    protected void loadPage() {
        TaskQuery query = new TaskQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setStatus(statusFilter);
        query.setTaskType(typeFilter);
        TaskAPI.getTaskPage(query, RepositoryAdapters.wrap(withLoading(data ->
                onPageLoaded(data.getList(), data.getTotal()))));
    }

    /**
     * 创建任务
     */
    public void createTask(TaskCreateForm form) {
        TaskAPI.createTask(form, RepositoryAdapters.wrap(withLoading(task -> {
            createdTask.postValue(task);
            operationResult.postValue("任务创建成功: " + task.getTaskId());
            loadTasks();
        }, msg -> error.postValue("创建失败: " + msg))));
    }

    /**
     * 取消任务
     */
    public void cancelTask(String taskId) {
        TaskAPI.cancelTask(taskId, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("任务已取消");
            loadTasks();
        }, msg -> error.postValue("取消失败: " + msg))));
    }

    /**
     * 下载任务结果
     */
    public void downloadTaskFile(String taskId) {
        taskRepository.downloadTaskFile(taskId, withLoading(v ->
                operationResult.postValue("下载成功，已保存到下载目录"),
                msg -> error.postValue("下载失败: " + msg)));
    }

    /**
     * 查看任务详情
     */
    public void getTaskDetail(String taskId) {
        TaskAPI.getTask(taskId, RepositoryAdapters.wrap(withLoading(taskDetail::postValue,
                msg -> error.postValue("查询详情失败: " + msg))));
    }

    public LiveData<List<TaskVO>> getTaskList() {
        return itemList;
    }

    public LiveData<TaskVO> getTaskDetail() {
        return taskDetail;
    }

    public LiveData<TaskVO> getCreatedTask() {
        return createdTask;
    }

    public long getTotal() {
        return total;
    }
}
