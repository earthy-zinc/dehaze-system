import {AxiosError} from "axios";
import {
    AuthAPI,
    CaptchaResult,
    configJavaAxios,
    LoginData,
    UserAPI,
    UserForm,
    UserInfo,
    UserQuery,
} from "dehaze-sdk-js";

(global as any).localStorage = {
    store: {} as Record<string, string>,
    getItem: function (key: string) {
        return this.store[key] || null;
    },
    setItem: function (key: string, value: string) {
        this.store[key] = value.toString();
    },
    removeItem: function (key: string) {
        delete this.store[key];
    },
    clear: function () {
        this.store = {};
    },
};
// 初始化服务
configJavaAxios({
    onRequest: (config) => {
        return {
            ...config,
            baseURL: "http://localhost:8080",
        };
    },
    onResponseError: (error: AxiosError) => {
        console.log("响应错误:\n", error.response?.data);
    },
});

interface TestResult {
    testName: string;
    success: boolean;
    message?: string;
    error?: any;
}

async function runUserAPITests(accessToken: string): Promise<TestResult[]> {
    console.log("开始测试 UserAPI...\n");
    const testResults: TestResult[] = [];

    try {
        configJavaAxios({
            onRequest: (config: any) => {
                return {
                    ...config,
                    baseURL: "http://localhost:8080",
                    headers: {
                        ...config.headers,
                        Authorization: `Bearer ${accessToken}`,
                    },
                };
            },
            onResponseError: (error) => {
                console.error("响应错误:", error);
            },
        });

        // 测试获取当前用户信息
        console.log("1. 测试获取当前用户信息:");
        const userInfo: UserInfo = await UserAPI.getInfo();
        console.log("   用户信息获取成功:", userInfo.nickname);
        testResults.push({testName: "获取当前用户信息", success: true});

        // 测试获取用户分页列表
        console.log("\n2. 测试获取用户分页列表:");
        const queryParams: UserQuery = {
            pageNum: 1,
            pageSize: 10,
            keywords: "",
            status: undefined,
            deptId: undefined,
            startTime: undefined,
            endTime: undefined,
        };
        const userPage = await UserAPI.getPage(queryParams);
        console.log("   用户分页数据获取成功，共", userPage.total, "条记录");
        testResults.push({testName: "获取用户分页列表", success: true});

        // 测试获取用户表单详情
        console.log("\n3. 测试获取用户表单详情:");
        const userForm = await UserAPI.getFormData(1);
        console.log("   用户表单数据获取成功，用户名:", userForm.username);
        testResults.push({testName: "获取用户表单详情", success: true});

        // 测试添加用户
        console.log("\n4. 测试添加用户:");
        const newUser: UserForm = {
            username: "newuser",
            nickname: "New User",
            mobile: "13800138002",
            gender: 1,
            deptId: 2,
            roleIds: [2],
            status: 1,
        };
        const addResult = await UserAPI.add(newUser);
        console.log("   用户添加成功");
        testResults.push({testName: "添加用户", success: true});

        // 测试修改用户
        console.log("\n5. 测试修改用户:");
        const updateResult = await UserAPI.update(1, {nickname: "Updated User"});
        console.log("   用户信息更新成功");
        testResults.push({testName: "修改用户", success: true});

        // 测试修改用户密码
        console.log("\n6. 测试修改用户密码:");
        const passwordResult = await UserAPI.updatePassword(1, "newpassword123");
        console.log("   用户密码修改结果:", passwordResult.status);
        testResults.push({testName: "修改用户密码", success: true});

        // 测试删除用户
        console.log("\n7. 测试删除用户:");
        const deleteResult = await UserAPI.deleteByIds("1,2");
        console.log("   用户删除成功，结果:", deleteResult.status);
        testResults.push({testName: "删除用户", success: true});
    } catch (error: any) {
        console.error("UserAPI测试过程中出现错误:", error.message);
        testResults.push({
            testName: "UserAPI整体测试",
            success: false,
            error: error.message,
        });
    }

    return testResults;
}

async function runAuthAPITests(): Promise<{
    results: TestResult[];
    accessToken?: string;
}> {
    console.log("\n开始测试 AuthAPI...\n");
    const testResults: TestResult[] = [];
    let accessToken: string | undefined;

    try {
        // 测试获取验证码
        console.log("1. 测试获取验证码:");
        const captchaResult: CaptchaResult = await AuthAPI.getCaptcha();
        console.log("   验证码获取成功，key:", captchaResult.captchaKey);
        testResults.push({testName: "获取验证码", success: true});

        // 测试登录
        console.log("\n2. 测试登录:");
        const loginData: LoginData = {
            username: "testuser",
            password: "testpass",
        };
        const loginResult = await AuthAPI.login(loginData);
        console.log("   登录成功，token类型:", loginResult.tokenType);
        accessToken = loginResult.accessToken;
        testResults.push({testName: "登录", success: true});

        // 测试注销
        console.log("\n3. 测试注销:");
        const logoutResult = await AuthAPI.logout();
        console.log("   注销成功");
        testResults.push({testName: "注销", success: true});
    } catch (error: any) {
        console.error("AuthAPI测试过程中出现错误:", error.message);
        testResults.push({
            testName: "AuthAPI整体测试",
            success: false,
            error: error.message,
        });
    }

    return {results: testResults, accessToken};
}

async function runTests(): Promise<void> {
    console.log("开始测试 Dehaze SDK...\n");

    const authTestResults = await runAuthAPITests();
    let allTestResults = [...authTestResults.results];

    if (authTestResults.accessToken) {
        const userTestResults = await runUserAPITests(authTestResults.accessToken);
        allTestResults = [...allTestResults, ...userTestResults];
    }

    // 输出测试结果摘要
    console.log("\n=== 测试结果摘要 ===");
    allTestResults.forEach((result, index) => {
        const status = result.success ? "✅ 通过" : "❌ 失败";
        console.log(`${index + 1}. ${result.testName}: ${status}`);
        if (!result.success && result.error) {
            console.log(`   错误信息: ${result.error}`);
        }
    });

    const passedTests = allTestResults.filter((r) => r.success).length;
    const totalTests = allTestResults.length;
    console.log(`\n总计: ${passedTests}/${totalTests} 个测试通过`);
}

// 运行测试
runTests();
