"use strict";
Object.defineProperty(exports, "__esModule", {value: true});
const dehaze_sdk_js_1 = require("dehaze-sdk-js");
// Mock server setup
const mockUserInfo = {
    userId: 1,
    username: 'testuser',
    nickname: 'Test User',
    avatar: 'https://example.com/avatar.jpg',
    roles: ['admin'],
    perms: ['user:list', 'user:create', 'user:update', 'user:delete']
};
const mockUserPage = {
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
            createTime: new Date().toISOString()
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
            createTime: new Date().toISOString()
        }
    ],
    total: 2
};
const mockUserForm = {
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
describe('UserAPI', () => {
    beforeAll(() => {
        // Setup mock server or intercept HTTP requests
        // This would typically use something like msw or nock
    });
    it('should get user info', async () => {
        // Mock the HTTP request
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '操作成功',
                data: mockUserInfo
            })
        });
        const userInfo = await dehaze_sdk_js_1.UserAPI.getInfo();
        expect(userInfo).toEqual(mockUserInfo);
    });
    it('should get user page', async () => {
        const queryParams = {
            pageNum: 1,
            pageSize: 10
        };
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '操作成功',
                data: mockUserPage
            })
        });
        const userPage = await dehaze_sdk_js_1.UserAPI.getPage(queryParams);
        expect(userPage).toEqual(mockUserPage);
    });
    it('should get user form data', async () => {
        const userId = 1;
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '操作成功',
                data: mockUserForm
            })
        });
        const userForm = await dehaze_sdk_js_1.UserAPI.getFormData(userId);
        expect(userForm).toEqual(mockUserForm);
    });
    it('should add user', async () => {
        const newUser = {
            username: 'newuser',
            nickname: 'New User',
            mobile: '13800138002',
            gender: 1,
            deptId: 2,
            roleIds: [2],
            status: 1,
        };
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '操作成功',
                data: {
                    id: 3,
                    ...newUser
                }
            })
        });
        const result = await dehaze_sdk_js_1.UserAPI.add(newUser);
        expect(result.data).toEqual({
            id: 3,
            ...newUser
        });
    });
    it('should update user', async () => {
        const userId = 1;
        const updateData = {
            nickname: 'Updated User'
        };
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '操作成功',
                data: {
                    id: userId,
                    ...updateData
                }
            })
        });
        const result = await dehaze_sdk_js_1.UserAPI.update(userId, updateData);
        expect(result.data).toEqual({
            id: userId,
            ...updateData
        });
    });
    it('should update user password', async () => {
        const userId = 1;
        const newPassword = 'newpassword123';
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '密码修改成功',
                data: null
            })
        });
        const result = await dehaze_sdk_js_1.UserAPI.updatePassword(userId, newPassword);
        expect(result.code).toBe('00000');
        expect(result.msg).toBe('密码修改成功');
    });
    it('should delete users by ids', async () => {
        const ids = '1,2';
        jest.spyOn(global, 'fetch').mockResolvedValue({
            json: jest.fn().mockResolvedValue({
                code: '00000',
                msg: '删除成功',
                data: null
            })
        });
        const result = await dehaze_sdk_js_1.UserAPI.deleteByIds(ids);
        expect(result.code).toBe('00000');
        expect(result.msg).toBe('删除成功');
    });
});
//# sourceMappingURL=user-api.test.js.map
