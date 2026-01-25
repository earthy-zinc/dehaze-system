create table dehaze_test.sys_task
(
    id              bigint auto_increment comment '主键ID'
        primary key,
    task_id         varchar(64)                           not null comment '任务ID（UUID）',
    task_type       varchar(32)                           not null comment '任务类型',
    status          varchar(32) default 'pending'         not null comment '任务状态',
    progress        int         default 0                 null comment '任务进度（百分比）',
    total_files     int         default 0                 null comment '总文件数',
    processed_files int         default 0                 null comment '已处理文件数',
    params          text                                  null comment '任务参数（JSON）',
    result          text                                  null comment '任务结果（下载链接）',
    error_message   text                                  null comment '错误信息',
    created_by      bigint                                null comment '创建人ID',
    created_at      datetime    default CURRENT_TIMESTAMP not null comment '创建时间',
    started_at      datetime                              null comment '开始时间',
    completed_at    datetime                              null comment '完成时间',
    expires_at      datetime                              null comment '过期时间',
    constraint idx_task_id
        unique (task_id)
);

create index idx_created_at
    on dehaze_test.sys_task (created_at);

create index idx_created_by
    on dehaze_test.sys_task (created_by);

create index idx_status
    on dehaze_test.sys_task (status);

