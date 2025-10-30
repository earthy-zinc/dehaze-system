import bodyParser from 'body-parser';
import express, {Application, Request, Response} from 'express';

const app: Application = express();
const port: number = 8080;

// 添加body parser中间件来处理FormData
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));

// 模拟用户信息
const mockUserInfo = {
    userId: 1,
    username: 'testuser',
    nickname: 'Test User',
    avatar: 'https://example.com/avatar.jpg',
    roles: ['admin'],
    perms: ['user:list', 'user:create', 'user:update', 'user:delete']
};

// 模拟用户分页数据
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

// 模拟用户表单数据
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

// 模拟验证码数据
const mockCaptchaResult = {
    captchaKey: 'mock-captcha-key',
    captchaBase64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=='
};

// 模拟登录结果
const mockLoginResult = {
    accessToken: 'mock-access-token',
    expires: 3600000,
    refreshToken: 'mock-refresh-token',
    tokenType: 'Bearer'
};

// 获取当前用户信息接口
app.get('/api/v1/users/me', (req: Request, res: Response) => {
    // 检查是否有认证头
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({
            code: '401',
            msg: '未授权访问',
            data: null
        });
    }

    return res.json({
        code: '00000',
        msg: '操作成功',
        data: mockUserInfo
    });
});

// 获取用户分页列表
app.get('/api/v1/users/page', (req: Request, res: Response) => {
    // 检查是否有认证头
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({
            code: '401',
            msg: '未授权访问',
            data: null
        });
    }

    return res.json({
        code: '00000',
        msg: '操作成功',
        data: mockUserPage
    });
});

// 获取用户表单详情
app.get('/api/v1/users/:userId/form', (req: Request, res: Response) => {
    // 检查是否有认证头
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({
            code: '401',
            msg: '未授权访问',
            data: null
        });
    }

    const userId: number = parseInt(req.params.userId);
    if (userId && userId > 0) {
        return res.json({
            code: '00000',
            msg: '操作成功',
            data: mockUserForm
        });
    } else {
        return res.status(400).json({
            code: '50000',
            msg: '参数错误',
            data: null
        });
    }
});

// 添加用户
app.post('/api/v1/users', (req: Request, res: Response) => {
    // 检查是否有认证头
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({
            code: '401',
            msg: '未授权访问',
            data: null
        });
    }

    const userData = req.body;
    if (userData.username && userData.nickname) {
        return res.json({
            code: '00000',
            msg: '操作成功',
            data: {
                id: 3,
                ...userData
            }
        });
    } else {
        return res.status(400).json({
            code: '50000',
            msg: '用户名和昵称不能为空',
            data: null
        });
    }
});

// 修改用户
app.put('/api/v1/users/:id', (req: Request, res: Response) => {
    // 检查是否有认证头
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({
            code: '401',
            msg: '未授权访问',
            data: null
        });
    }

    const userId: number = parseInt(req.params.id);
    const userData = req.body;

    if (userId && userId > 0 && userData.nickname) {
        return res.json({
            code: '00000',
            msg: '操作成功',
            data: {
                id: userId,
                ...userData
            }
        });
    } else {
        return res.status(400).json({
            code: '50000',
            msg: '参数错误',
            data: null
        });
    }
});

// 修改用户密码
app.patch('/api/v1/users/:id/password', (req: Request, res: Response) => {
    // 检查是否有认证头
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({
            code: '401',
            msg: '未授权访问',
            data: null
        });
    }

    const userId: number = parseInt(req.params.id);
    const {password} = req.query;

    if (userId && userId > 0 && password) {
        return res.json({
            code: '00000',
            msg: '密码修改成功',
            data: null
        });
    } else {
        return res.status(400).json({
            code: '50000',
            msg: '参数错误',
            data: null
        });
    }
});

// 删除用户
app.delete('/api/v1/users/:ids', (req: Request, res: Response) => {
    // 检查是否有认证头
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({
            code: '401',
            msg: '未授权访问',
            data: null
        });
    }

    const ids: string = req.params.ids;

    if (ids) {
        return res.json({
            code: '00000',
            msg: '删除成功',
            data: null
        });
    } else {
        return res.status(400).json({
            code: '50000',
            msg: '参数错误',
            data: null
        });
    }
});

// 登录接口
app.post('/api/v1/auth/login', (req: Request, res: Response) => {
    console.log('Received login request:', req.body);
    // 对于FormData，我们需要从req.body中获取数据
    const {username, password} = req.body;

    if (username && password) {
        return res.json({
            code: '00000',
            msg: '登录成功',
            data: mockLoginResult
        });
    } else {
        return res.status(400).json({
            code: '50000',
            msg: '用户名和密码不能为空',
            data: null
        });
    }
});

// 注销接口
app.delete('/api/v1/auth/logout', (req: Request, res: Response) => {
    return res.json({
        code: '00000',
        msg: '注销成功',
        data: null
    });
});

// 获取验证码接口
app.get('/api/v1/auth/captcha', (req: Request, res: Response) => {
    return res.json({
        code: '00000',
        msg: '操作成功',
        data: mockCaptchaResult
    });
});

// 启动服务器
app.listen(port, () => {
    console.log(`Mock server is running on http://localhost:${port}`);
});

export default app;
