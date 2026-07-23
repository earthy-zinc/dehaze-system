package com.pei.dehaze.ui.task.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.repository.TaskRepository;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.task.TaskCreateForm;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskStatus;
import com.pei.dehaze.sdk.model.task.TaskType;
import com.pei.dehaze.sdk.model.task.TaskVO;

import java.util.ArrayList;
import java.util.List;

/**
 * 任务管理 ViewModel
 */
public class TaskViewModel extends BaseViewModel {

    private final TaskRepository taskRepository;

    private final MutableLiveData<List<TaskVO>> taskList = new MutableLiveData<>();
    private final MutableLiveData<TaskVO> taskDetail = new MutableLiveData<>();
    private final MutableLiveData<TaskVO> createdTask = new MutableLiveData<>();

    private int pageNum = 1;
    private int pageSize = 10;
    private TaskStatus statusFilter;
    private TaskType typeFilter;
    private long total = 0;

    public TaskViewModel() {
        taskRepository = new TaskRepository();
    }

    /**
     * 加载任务列表（首页）
     */
    public void loadTasks() {
        pageNum = 1;
        fetchTasks();
    }

    /**
     * 按状态筛选
     */
    public void filterByStatus(TaskStatus status) {
        this.statusFilter = status;
        pageNum = 1;
        fetchTasks();
    }

    /**
     * 按类型筛选
     */
    public void filterByType(TaskType taskType) {
        this.typeFilter = taskType;
        pageNum = 1;
        fetchTasks();
    }

    /**
     * 加载下一页
     */
    public void loadMore() {
        if (taskList.getValue() == null || taskList.getValue().size() >= total) {
            return;
        }
        pageNum++;
        fetchTasks();
    }

    private void fetchTasks() {
        TaskQuery query = new TaskQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setStatus(statusFilter);
        query.setTaskType(typeFilter);
        taskRepository.getTasks(query, withLoading(data -> {
            List<TaskVO> tasks = data.getList();
            TaskViewModel.this.total = data.getTotal();
            if (pageNum == 1) {
                taskList.postValue(tasks);
            } else {
                List<TaskVO> current = taskList.getValue();
                if (current == null) {
                    current = new ArrayList<>();
                }
                current.addAll(tasks);
                taskList.postValue(current);
            }
        }));
    }

    /**
     * 创建任务
     */
    public void createTask(TaskCreateForm form) {
        taskRepository.createTask(form, withLoading(task -> {
            createdTask.postValue(task);
            operationResult.postValue("任务创建成功: " + task.getTaskId());
            loadTasks();
        }, msg -> error.postValue("创建失败: " + msg)));
    }

    /**
     * 取消任务
     */
    public void cancelTask(String taskId) {
        taskRepository.cancelTask(taskId, withLoading(v -> {
            operationResult.postValue("任务已取消");
            loadTasks();
        }, msg -> error.postValue("取消失败: " + msg)));
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
        taskRepository.getTask(taskId, withLoading(taskDetail::postValue,
                msg -> error.postValue("查询详情失败: " + msg)));
    }

    public LiveData<List<TaskVO>> getTaskList() {
        return taskList;
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
