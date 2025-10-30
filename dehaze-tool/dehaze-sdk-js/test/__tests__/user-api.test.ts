import {UserAPI, UserForm, UserInfo, UserPageVO, UserQuery} from 'dehaze-sdk-js';
import {PageResult} from 'dehaze-sdk-js/src/types';

// Mock server setup
const mockUserInfo: UserInfo = {
    userId: 1,
    username: 'testuser',
    nickname: 'Test User',
    avatar: 'https://example.com/avatar.jpg',
    roles: ['admin'],
    perms: ['user:list', 'user:create', 'user:update', 'user:delete']
};

const mockUserPage: PageResult<UserPageVO[]> = {
    list: [
        {
            id: 1,
            username: 'testuser',
            nickname: 'Test User',
            avatar: 'https://example.com/avatar.jpg',
            mobile: '13800138000',
            genderLabel: '男',
            deptName: '技术部',
            roleNames: '管理员',
            status: 1,
            createTime: new Date()
        },
        {
            id: 2,
            username: 'testuser2',
            nickname: 'Test User 2',
            avatar: 'https://example.com/avatar2.jpg',
            mobile: '13800138001',
            genderLabel: '女',
            deptName: '产品部',
            roleNames: '普通用户',
            status: 1,
            createTime: new Date()
        }
    ],
    total: 2
};

const mockUserForm: UserForm = {
    id: 1,
    username: 'testuser',
    nickname: 'Test User',
    avatar: 'https://example.com/avatar.jpg',
    mobile: '13800138000',
    gender: 1,
    deptId: 1,
    roleIds: [1],
    status: 1
};

// 简单的测试，只验证类型是否正确
describe('UserAPI Types', () => {
    it('should have correct type for UserInfo', () => {
        const userInfo: UserInfo = mockUserInfo;
        expect(userInfo.userId).toBe(1);
        expect(userInfo.username).toBe('testuser');
    });

    it('should have correct type for UserPageVO', () => {
        const userPage: PageResult<UserPageVO[]> = mockUserPage;
        expect(userPage.total).toBe(2);
        expect(userPage.list[0].id).toBe(1);
    });

    it('should have correct type for UserForm', () => {
        const userForm: UserForm = mockUserForm;
        expect(userForm.username).toBe('testuser');
        expect(userForm.deptId).toBe(1);
    });

    it('should have correct type for UserQuery', () => {
        const userQuery: UserQuery = {
            pageNum: 1,
            pageSize: 10,
            keywords: 'test',
            status: 1,
            deptId: 1
        };

        expect(userQuery.pageNum).toBe(1);
        expect(userQuery.keywords).toBe('test');
    });

    it('should have all required methods', () => {
        expect(typeof UserAPI.getInfo).toBe('function');
        expect(typeof UserAPI.getPage).toBe('function');
        expect(typeof UserAPI.getFormData).toBe('function');
        expect(typeof UserAPI.add).toBe('function');
        expect(typeof UserAPI.update).toBe('function');
        expect(typeof UserAPI.updatePassword).toBe('function');
        expect(typeof UserAPI.deleteByIds).toBe('function');
        expect(typeof UserAPI.downloadTemplate).toBe('function');
        expect(typeof UserAPI.export).toBe('function');
        expect(typeof UserAPI.import).toBe('function');
    });
});
