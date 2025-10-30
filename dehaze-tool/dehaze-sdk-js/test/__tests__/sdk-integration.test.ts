import {
    AlgorithmAPI,
    AuthAPI,
    CaptchaResult,
    DatasetAPI,
    DeptAPI,
    DictAPI,
    FileAPI,
    LoginData,
    MenuAPI,
    ModelAPI,
    RoleAPI,
    UserAPI,
    UserForm,
    UserInfo,
    UserPageVO
} from "dehaze-sdk-js";
import {PageResult} from "dehaze-sdk-js/src/types";

describe('Dehaze SDK Integration', () => {
    test('should have correct UserAPI types', () => {
        // 测试 UserAPI 类型
        const mockUserInfo: UserInfo = {
            userId: 1,
            username: "testuser",
            nickname: "Test User",
            avatar: "https://example.com/avatar.jpg",
            roles: ["admin"],
            perms: ["user:list", "user:create", "user:update", "user:delete"]
        };

        const mockUserPage: PageResult<UserPageVO[]> = {
            list: [
                {
                    id: 1,
                    username: "testuser",
                    nickname: "Test User",
                    avatar: "https://example.com/avatar.jpg",
                    mobile: "13800138000",
                    genderLabel: "男",
                    deptName: "技术部",
                    roleNames: "管理员",
                    status: 1,
                    createTime: new Date()
                }
            ],
            total: 1
        };

        const mockUserForm: UserForm = {
            id: 1,
            username: "testuser",
            nickname: "Test User",
            avatar: "https://example.com/avatar.jpg",
            mobile: "13800138000",
            gender: 1,
            deptId: 1,
            roleIds: [1],
            status: 1
        };

        expect(mockUserInfo.userId).toBe(1);
        expect(mockUserPage.total).toBe(1);
        expect(mockUserForm.username).toBe("testuser");
    });

    test('should have correct AuthAPI types', () => {
        // 测试 AuthAPI 类型
        const mockLoginData: LoginData = {
            username: "testuser",
            password: "testpass"
        };

        const mockCaptchaResult: CaptchaResult = {
            captchaKey: "mock-captcha-key",
            captchaBase64: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        };

        expect(mockLoginData.username).toBe("testuser");
        expect(mockCaptchaResult.captchaKey).toBe("mock-captcha-key");
    });

    test('should have UserAPI methods', () => {
        // 测试 API 方法存在
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

    test('should have AuthAPI methods', () => {
        expect(typeof AuthAPI.login).toBe('function');
        expect(typeof AuthAPI.logout).toBe('function');
        expect(typeof AuthAPI.getCaptcha).toBe('function');
    });

    test('should have all API modules with required methods', () => {
        // 测试所有 API 模块存在
        expect(AlgorithmAPI).toBeDefined();
        expect(DatasetAPI).toBeDefined();
        expect(DeptAPI).toBeDefined();
        expect(DictAPI).toBeDefined();
        expect(FileAPI).toBeDefined();
        expect(MenuAPI).toBeDefined();
        expect(ModelAPI).toBeDefined();
        expect(RoleAPI).toBeDefined();
    });
});
