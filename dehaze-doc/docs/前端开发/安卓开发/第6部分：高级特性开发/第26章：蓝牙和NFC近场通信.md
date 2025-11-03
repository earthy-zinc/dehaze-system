# 第26章：蓝牙和NFC近场通信

蓝牙和NFC（Near Field Communication，近场通信）是Android设备重要的短距离无线通信技术。蓝牙适用于设备间的持续数据传输，而NFC主要用于快速配对和支付等场景。本章将详细介绍这两种技术的开发方法和最佳实践。

## 26.1 蓝牙技术概述

Android支持多种蓝牙技术，包括传统蓝牙、低功耗蓝牙（BLE）和蓝牙音频。

### 26.1.1 蓝牙技术分类

```java
public class BluetoothTechnologyTypes {
    // 蓝牙类型
    public static final String BLUETOOTH_CLASSIC = "经典蓝牙";
    public static final String BLUETOOTH_LE = "低功耗蓝牙";
    public static final String BLUETOOTH_DUAL_MODE = "双模蓝牙";

    // 蓝牙配置文件
    public static final String PROFILE_A2DP = "高级音频分发配置文件";
    public static final String PROFILE_HFP = "免提配置文件";
    public static final String PROFILE_HID = "人机接口设备配置文件";
    public static final String PROFILE_PBAP = "电话簿访问配置文件";
    public static final String PROFILE_MAP = "消息访问配置文件";

    // BLE GATT服务
    public static final String SERVICE_GENERIC_ACCESS = "通用访问服务";
    public static final String SERVICE_GENERIC_ATTRIBUTE = "通用属性服务";
    public static final String SERVICE_DEVICE_INFORMATION = "设备信息服务";
    public static final String SERVICE_BATTERY = "电池服务";
    public static final String SERVICE_HEART_RATE = "心率服务";
    public static final String SERVICE_ENVIRONMENTAL_SENSING = "环境传感服务";

    // 权限常量
    public static final String PERMISSION_BLUETOOTH = Manifest.permission.BLUETOOTH;
    public static final String PERMISSION_BLUETOOTH_ADMIN = Manifest.permission.BLUETOOTH_ADMIN;
    public static final String PERMISSION_BLUETOOTH_SCAN = Manifest.permission.BLUETOOTH_SCAN;
    public static final String PERMISSION_BLUETOOTH_ADVERTISE = Manifest.permission.BLUETOOTH_ADVERTISE;
    public static final String PERMISSION_BLUETOOTH_CONNECT = Manifest.permission.BLUETOOTH_CONNECT;
    public static final String PERMISSION_ACCESS_FINE_LOCATION = Manifest.permission.ACCESS_FINE_LOCATION;
}
```

### 26.1.2 蓝牙适配器管理

蓝牙适配器是蓝牙开发的核心组件：

```java
public class BluetoothAdapterManager {
    private static final String TAG = "BluetoothAdapterManager";
    private static final int REQUEST_ENABLE_BLUETOOTH = 1001;

    private Context context;
    private BluetoothAdapter bluetoothAdapter;
    private BluetoothCallback callback;

    public interface BluetoothCallback {
        void onBluetoothEnabled();
        void onBluetoothDisabled();
        void onBluetoothError(String error);
        void onPermissionDenied();
    }

    public BluetoothAdapterManager(Context context, BluetoothCallback callback) {
        this.context = context;
        this.callback = callback;
        initializeBluetooth();
    }

    /**
     * 初始化蓝牙适配器
     */
    private void initializeBluetooth() {
        // Android 12+ 使用BluetoothManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            BluetoothManager bluetoothManager = (BluetoothManager) context.getSystemService(Context.BLUETOOTH_SERVICE);
            if (bluetoothManager != null) {
                bluetoothAdapter = bluetoothManager.getAdapter();
            }
        } else {
            // 旧版本使用getDefaultAdapter
            bluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
        }

        if (bluetoothAdapter == null) {
            Log.e(TAG, "设备不支持蓝牙");
            if (callback != null) {
                callback.onBluetoothError("设备不支持蓝牙");
            }
        }
    }

    /**
     * 检查蓝牙是否支持
     */
    public boolean isBluetoothSupported() {
        return bluetoothAdapter != null;
    }

    /**
     * 检查蓝牙是否启用
     */
    public boolean isBluetoothEnabled() {
        return bluetoothAdapter != null && bluetoothAdapter.isEnabled();
    }

    /**
     * 检查蓝牙权限
     */
    public boolean hasBluetoothPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // Android 12+ 需要新的蓝牙权限
            return ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED &&
                   ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED;
        } else {
            // 旧版本权限
            return ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH) == PackageManager.PERMISSION_GRANTED &&
                   ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_ADMIN) == PackageManager.PERMISSION_GRANTED;
        }
    }

    /**
     * 请求启用蓝牙
     */
    public void requestEnableBluetooth(Activity activity) {
        if (!isBluetoothSupported()) {
            if (callback != null) {
                callback.onBluetoothError("设备不支持蓝牙");
            }
            return;
        }

        if (!hasBluetoothPermissions()) {
            if (callback != null) {
                callback.onPermissionDenied();
            }
            return;
        }

        if (!bluetoothAdapter.isEnabled()) {
            Intent enableBtIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
            activity.startActivityForResult(enableBtIntent, REQUEST_ENABLE_BLUETOOTH);
        } else {
            if (callback != null) {
                callback.onBluetoothEnabled();
            }
        }
    }

    /**
     * 启用蓝牙（无需用户交互）
     */
    public boolean enableBluetooth() {
        if (bluetoothAdapter != null && !bluetoothAdapter.isEnabled()) {
            return bluetoothAdapter.enable();
        }
        return false;
    }

    /**
     * 禁用蓝牙
     */
    public boolean disableBluetooth() {
        if (bluetoothAdapter != null && bluetoothAdapter.isEnabled()) {
            return bluetoothAdapter.disable();
        }
        return false;
    }

    /**
     * 获取蓝牙状态信息
     */
    public String getBluetoothStatus() {
        if (!isBluetoothSupported()) {
            return "设备不支持蓝牙";
        }

        if (!hasBluetoothPermissions()) {
            return "缺少蓝牙权限";
        }

        StringBuilder status = new StringBuilder();
        status.append("蓝牙状态: ").append(bluetoothAdapter.isEnabled() ? "已启用" : "已禁用").append("\n");
        status.append("设备名称: ").append(bluetoothAdapter.getName()).append("\n");
        status.append("设备地址: ").append(bluetoothAdapter.getAddress()).append("\n");
        status.append("扫描模式: ").append(getScanModeDescription(bluetoothAdapter.getScanMode())).append("\n");
        status.append("状态: ").append(getStateDescription(bluetoothAdapter.getState())).append("\n");

        return status.toString();
    }

    private String getScanModeDescription(int scanMode) {
        switch (scanMode) {
            case BluetoothAdapter.SCAN_MODE_CONNECTABLE_DISCOVERABLE:
                return "可发现且可连接";
            case BluetoothAdapter.SCAN_MODE_CONNECTABLE:
                return "可连接但不可发现";
            case BluetoothAdapter.SCAN_MODE_NONE:
                return "不可发现且不可连接";
            default:
                return "未知";
        }
    }

    private String getStateDescription(int state) {
        switch (state) {
            case BluetoothAdapter.STATE_OFF:
                return "关闭";
            case BluetoothAdapter.STATE_TURNING_ON:
                return "正在开启";
            case BluetoothAdapter.STATE_ON:
                return "开启";
            case BluetoothAdapter.STATE_TURNING_OFF:
                return "正在关闭";
            default:
                return "未知";
        }
    }

    /**
     * 处理Activity结果
     */
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_ENABLE_BLUETOOTH) {
            if (resultCode == Activity.RESULT_OK) {
                Log.d(TAG, "蓝牙已启用");
                if (callback != null) {
                    callback.onBluetoothEnabled();
                }
            } else {
                Log.d(TAG, "蓝牙启用被拒绝");
                if (callback != null) {
                    callback.onBluetoothDisabled();
                }
            }
        }
    }

    /**
     * 获取蓝牙适配器
     */
    public BluetoothAdapter getBluetoothAdapter() {
        return bluetoothAdapter;
    }
}
```

## 26.2 蓝牙设备发现和连接

### 26.2.1 设备发现

设备发现是蓝牙开发的基础功能：

```java
public class BluetoothDeviceDiscovery {
    private static final String TAG = "BluetoothDeviceDiscovery";
    private static final long DISCOVERY_TIMEOUT = 12000; // 12秒发现超时

    private Context context;
    private BluetoothAdapter bluetoothAdapter;
    private DeviceDiscoveryCallback callback;
    private List<BluetoothDevice> discoveredDevices;
    private BroadcastReceiver discoveryReceiver;
    private Handler timeoutHandler;
    private boolean isDiscovering = false;

    public interface DeviceDiscoveryCallback {
        void onDeviceDiscovered(BluetoothDevice device, int rssi, byte[] scanRecord);
        void onDiscoveryStarted();
        void onDiscoveryFinished();
        void onDiscoveryError(String error);
        void onBluetoothStateChanged(int state);
    }

    public BluetoothDeviceDiscovery(Context context, BluetoothAdapter adapter, DeviceDiscoveryCallback callback) {
        this.context = context;
        this.bluetoothAdapter = adapter;
        this.callback = callback;
        this.discoveredDevices = new ArrayList<>();
        this.timeoutHandler = new Handler(Looper.getMainLooper());

        initializeDiscoveryReceiver();
    }

    /**
     * 初始化发现广播接收器
     */
    private void initializeDiscoveryReceiver() {
        discoveryReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();

                if (BluetoothAdapter.ACTION_DISCOVERY_STARTED.equals(action)) {
                    onDiscoveryStarted();
                } else if (BluetoothAdapter.ACTION_DISCOVERY_FINISHED.equals(action)) {
                    onDiscoveryFinished();
                } else if (BluetoothDevice.ACTION_FOUND.equals(action)) {
                    onDeviceFound(intent);
                } else if (BluetoothAdapter.ACTION_STATE_CHANGED.equals(action)) {
                    onBluetoothStateChanged(intent);
                }
            }
        };

        IntentFilter filter = new IntentFilter();
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_STARTED);
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED);
        filter.addAction(BluetoothDevice.ACTION_FOUND);
        filter.addAction(BluetoothAdapter.ACTION_STATE_CHANGED);

        context.registerReceiver(discoveryReceiver, filter);
    }

    /**
     * 开始设备发现
     */
    public boolean startDiscovery() {
        if (bluetoothAdapter == null || !bluetoothAdapter.isEnabled()) {
            if (callback != null) {
                callback.onDiscoveryError("蓝牙未启用");
            }
            return false;
        }

        if (bluetoothAdapter.isDiscovering()) {
            bluetoothAdapter.cancelDiscovery();
        }

        discoveredDevices.clear();
        boolean started = bluetoothAdapter.startDiscovery();

        if (started) {
            isDiscovering = true;
            // 设置发现超时
            timeoutHandler.postDelayed(this::stopDiscovery, DISCOVERY_TIMEOUT);
        } else {
            if (callback != null) {
                callback.onDiscoveryError("无法开始设备发现");
            }
        }

        return started;
    }

    /**
     * 停止设备发现
     */
    public void stopDiscovery() {
        if (bluetoothAdapter != null && bluetoothAdapter.isDiscovering()) {
            bluetoothAdapter.cancelDiscovery();
        }

        timeoutHandler.removeCallbacksAndMessages(null);
        isDiscovering = false;
    }

    /**
     * 发现开始处理
     */
    private void onDiscoveryStarted() {
        Log.d(TAG, "设备发现已开始");
        if (callback != null) {
            callback.onDiscoveryStarted();
        }
    }

    /**
     * 发现完成处理
     */
    private void onDiscoveryFinished() {
        Log.d(TAG, "设备发现已完成，发现设备数: " + discoveredDevices.size());
        isDiscovering = false;
        timeoutHandler.removeCallbacksAndMessages(null);

        if (callback != null) {
            callback.onDiscoveryFinished();
        }
    }

    /**
     * 设备发现处理
     */
    private void onDeviceFound(Intent intent) {
        BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
        short rssi = intent.getShortExtra(BluetoothDevice.EXTRA_RSSI, Short.MIN_VALUE);
        byte[] scanRecord = intent.getByteArrayExtra(BluetoothDevice.EXTRA_CLASS);

        if (device != null && !discoveredDevices.contains(device)) {
            discoveredDevices.add(device);

            Log.d(TAG, String.format("发现设备: %s (%s), RSSI: %d dBm",
                device.getName(), device.getAddress(), rssi));

            if (callback != null) {
                callback.onDeviceDiscovered(device, rssi, scanRecord);
            }
        }
    }

    /**
     * 蓝牙状态变化处理
     */
    private void onBluetoothStateChanged(Intent intent) {
        int state = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, BluetoothAdapter.ERROR);
        Log.d(TAG, "蓝牙状态变化: " + state);

        if (callback != null) {
            callback.onBluetoothStateChanged(state);
        }

        // 如果蓝牙关闭，停止发现
        if (state == BluetoothAdapter.STATE_OFF) {
            stopDiscovery();
        }
    }

    /**
     * 获取已发现的设备列表
     */
    public List<BluetoothDevice> getDiscoveredDevices() {
        return new ArrayList<>(discoveredDevices);
    }

    /**
     * 获取已配对的设备列表
     */
    public Set<BluetoothDevice> getPairedDevices() {
        if (bluetoothAdapter != null) {
            return bluetoothAdapter.getBondedDevices();
        }
        return new HashSet<>();
    }

    /**
     * 检查设备是否已配对
     */
    public boolean isDevicePaired(BluetoothDevice device) {
        Set<BluetoothDevice> pairedDevices = getPairedDevices();
        return pairedDevices.contains(device);
    }

    /**
     * 获取设备详细信息
     */
    public String getDeviceInfo(BluetoothDevice device) {
        if (device == null) return "设备信息不可用";

        StringBuilder info = new StringBuilder();
        info.append("设备名称: ").append(device.getName()).append("\n");
        info.append("设备地址: ").append(device.getAddress()).append("\n");
        info.append("设备类型: ").append(getDeviceTypeDescription(device.getType())).append("\n");
        info.append("绑定状态: ").append(getBondStateDescription(device.getBondState())).append("\n");

        BluetoothClass bluetoothClass = device.getBluetoothClass();
        if (bluetoothClass != null) {
            info.append("设备类别: ").append(bluetoothClass.getDeviceClass()).append("\n");
            info.append("主要服务: ").append(getMajorServiceDescription(bluetoothClass.getMajorDeviceClass())).append("\n");
        }

        return info.toString();
    }

    private String getDeviceTypeDescription(int type) {
        switch (type) {
            case BluetoothDevice.DEVICE_TYPE_CLASSIC:
                return "经典蓝牙";
            case BluetoothDevice.DEVICE_TYPE_LE:
                return "低功耗蓝牙";
            case BluetoothDevice.DEVICE_TYPE_DUAL:
                return "双模蓝牙";
            default:
                return "未知";
        }
    }

    private String getBondStateDescription(int bondState) {
        switch (bondState) {
            case BluetoothDevice.BOND_NONE:
                return "未配对";
            case BluetoothDevice.BOND_BONDING:
                return "配对中";
            case BluetoothDevice.BOND_BONDED:
                return "已配对";
            default:
                return "未知";
        }
    }

    private String getMajorServiceDescription(int majorClass) {
        switch (majorClass) {
            case BluetoothClass.Device.Major.AUDIO_VIDEO:
                return "音频/视频";
            case BluetoothClass.Device.Major.COMPUTER:
                return "计算机";
            case BluetoothClass.Device.Major.HEALTH:
                return "健康设备";
            case BluetoothClass.Device.Major.IMAGING:
                return "图像设备";
            case BluetoothClass.Device.Major.MISC:
                return "杂项设备";
            case BluetoothClass.Device.Major.NETWORKING:
                return "网络设备";
            case BluetoothClass.Device.Major.PERIPHERAL:
                return "外围设备";
            case BluetoothClass.Device.Major.PHONE:
                return "电话设备";
            case BluetoothClass.Device.Major.TOY:
                return "玩具设备";
            case BluetoothClass.Device.Major.UNCATEGORIZED:
                return "未分类设备";
            case BluetoothClass.Device.Major.WEARABLE:
                return "可穿戴设备";
            default:
                return "未知设备";
        }
    }

    /**
     * 按信号强度排序设备
     */
    public void sortDevicesByRSSI(List<BluetoothDevice> devices, Map<BluetoothDevice, Integer> rssiMap) {
        devices.sort((d1, d2) -> {
            Integer rssi1 = rssiMap.get(d1);
            Integer rssi2 = rssiMap.get(d2);
            if (rssi1 != null && rssi2 != null) {
                return rssi2.compareTo(rssi1); // 降序排序，信号强的在前
            }
            return 0;
        });
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        stopDiscovery();
        if (discoveryReceiver != null) {
            context.unregisterReceiver(discoveryReceiver);
        }
        timeoutHandler.removeCallbacksAndMessages(null);
    }
}
```

### 26.2.2 设备配对和连接

设备配对和连接是蓝牙通信的基础：

```java
public class BluetoothConnectionManager {
    private static final String TAG = "BluetoothConnectionManager";
    private static final int CONNECT_TIMEOUT = 10000; // 10秒连接超时

    private Context context;
    private BluetoothAdapter bluetoothAdapter;
    private ConnectionCallback callback;
    private BroadcastReceiver pairingReceiver;
    private Handler connectHandler;
    private Map<String, BluetoothSocket> connectedSockets;
    private Map<String, Thread> connectionThreads;

    public interface ConnectionCallback {
        void onPairingStarted(BluetoothDevice device);
        void onPairingFinished(BluetoothDevice device, boolean success);
        void onConnecting(BluetoothDevice device);
        void onConnected(BluetoothDevice device, BluetoothSocket socket);
        void onDisconnected(BluetoothDevice device);
        void onConnectionError(BluetoothDevice device, String error);
        void onDataReceived(BluetoothDevice device, byte[] data);
    }

    public BluetoothConnectionManager(Context context, BluetoothAdapter adapter, ConnectionCallback callback) {
        this.context = context;
        this.bluetoothAdapter = adapter;
        this.callback = callback;
        this.connectedSockets = new HashMap<>();
        this.connectionThreads = new HashMap<>();
        this.connectHandler = new Handler(Looper.getMainLooper());

        initializePairingReceiver();
    }

    /**
     * 初始化配对广播接收器
     */
    private void initializePairingReceiver() {
        pairingReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();
                BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);

                if (device == null) return;

                if (BluetoothDevice.ACTION_BOND_STATE_CHANGED.equals(action)) {
                    onBondStateChanged(device, intent);
                } else if (BluetoothDevice.ACTION_PAIRING_REQUEST.equals(action)) {
                    onPairingRequest(device, intent);
                }
            }
        };

        IntentFilter filter = new IntentFilter();
        filter.addAction(BluetoothDevice.ACTION_BOND_STATE_CHANGED);
        filter.addAction(BluetoothDevice.ACTION_PAIRING_REQUEST);

        context.registerReceiver(pairingReceiver, filter);
    }

    /**
     * 开始设备配对
     */
    public boolean startPairing(BluetoothDevice device) {
        if (device == null) {
            Log.e(TAG, "设备为空");
            return false;
        }

        if (device.getBondState() == BluetoothDevice.BOND_BONDED) {
            Log.d(TAG, "设备已配对: " + device.getAddress());
            return true;
        }

        if (device.getBondState() == BluetoothDevice.BOND_BONDING) {
            Log.d(TAG, "设备正在配对中: " + device.getAddress());
            return true;
        }

        try {
            boolean result = device.createBond();
            if (result) {
                Log.d(TAG, "开始配对设备: " + device.getAddress());
                if (callback != null) {
                    callback.onPairingStarted(device);
                }
            } else {
                Log.e(TAG, "无法开始配对设备: " + device.getAddress());
            }
            return result;
        } catch (Exception e) {
            Log.e(TAG, "配对异常: " + e.getMessage());
            return false;
        }
    }

    /**
     * 取消配对
     */
    public boolean cancelPairing(BluetoothDevice device) {
        if (device != null && device.getBondState() == BluetoothDevice.BOND_BONDING) {
            try {
                // 通过反射调用取消配对方法
                Method method = device.getClass().getMethod("cancelBondProcess");
                return (Boolean) method.invoke(device);
            } catch (Exception e) {
                Log.e(TAG, "取消配对失败: " + e.getMessage());
            }
        }
        return false;
    }

    /**
     * 连接设备
     */
    public void connectDevice(BluetoothDevice device) {
        if (device == null) {
            Log.e(TAG, "设备为空");
            return;
        }

        // 如果已连接，直接返回
        if (connectedSockets.containsKey(device.getAddress())) {
            Log.d(TAG, "设备已连接: " + device.getAddress());
            return;
        }

        // 检查配对状态
        if (device.getBondState() != BluetoothDevice.BOND_BONDED) {
            Log.w(TAG, "设备未配对，先进行配对: " + device.getAddress());
            startPairing(device);
            return;
        }

        // 停止发现（连接时需要停止发现）
        if (bluetoothAdapter.isDiscovering()) {
            bluetoothAdapter.cancelDiscovery();
        }

        if (callback != null) {
            callback.onConnecting(device);
        }

        // 在后台线程中连接
        Thread connectThread = new Thread(() -> {
            BluetoothSocket socket = null;
            try {
                // 获取UUID
                UUID uuid = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB"); // SPP UUID
                socket = device.createRfcommSocketToServiceRecord(uuid);

                // 设置连接超时
                socket.connect();

                // 连接成功
                handleConnectionSuccess(device, socket);

            } catch (IOException e) {
                Log.e(TAG, "连接失败: " + e.getMessage());

                // 尝试备用连接方法
                socket = fallbackConnection(device);
                if (socket != null && socket.isConnected()) {
                    handleConnectionSuccess(device, socket);
                } else {
                    handleConnectionError(device, "连接失败: " + e.getMessage());
                }
            }
        });

        connectionThreads.put(device.getAddress(), connectThread);
        connectThread.start();

        // 设置连接超时
        connectHandler.postDelayed(() -> {
            if (connectionThreads.containsKey(device.getAddress())) {
                handleConnectionError(device, "连接超时");
            }
        }, CONNECT_TIMEOUT);
    }

    /**
     * 备用连接方法
     */
    private BluetoothSocket fallbackConnection(BluetoothDevice device) {
        try {
            // 使用反射获取隐藏的socket
            Method method = device.getClass().getMethod("createRfcommSocket", int.class);
            BluetoothSocket socket = (BluetoothSocket) method.invoke(device, 1);
            socket.connect();
            return socket;
        } catch (Exception e) {
            Log.e(TAG, "备用连接失败: " + e.getMessage());
            return null;
        }
    }

    /**
     * 处理连接成功
     */
    private void handleConnectionSuccess(BluetoothDevice device, BluetoothSocket socket) {
        connectHandler.post(() -> {
            String address = device.getAddress();
            connectedSockets.put(address, socket);
            connectionThreads.remove(address);

            Log.d(TAG, "设备连接成功: " + address);

            if (callback != null) {
                callback.onConnected(device, socket);
            }

            // 开始数据接收
            startDataReceiver(device, socket);
        });
    }

    /**
     * 处理连接错误
     */
    private void handleConnectionError(BluetoothDevice device, String error) {
        connectHandler.post(() -> {
            String address = device.getAddress();
            Thread thread = connectionThreads.remove(address);

            if (thread != null) {
                thread.interrupt();
            }

            Log.e(TAG, "设备连接失败: " + address + ", 错误: " + error);

            if (callback != null) {
                callback.onConnectionError(device, error);
            }
        });
    }

    /**
     * 开始数据接收
     */
    private void startDataReceiver(BluetoothDevice device, BluetoothSocket socket) {
        Thread receiveThread = new Thread(() -> {
            InputStream inputStream = null;
            try {
                inputStream = socket.getInputStream();
                byte[] buffer = new byte[1024];
                int bytes;

                while (!Thread.currentThread().isInterrupted() && socket.isConnected()) {
                    bytes = inputStream.read(buffer);
                    if (bytes > 0) {
                        byte[] data = new byte[bytes];
                        System.arraycopy(buffer, 0, data, 0, bytes);

                        if (callback != null) {
                            callback.onDataReceived(device, data);
                        }
                    }
                }
            } catch (IOException e) {
                Log.e(TAG, "数据接收错误: " + e.getMessage());
                handleDisconnection(device);
            } finally {
                if (inputStream != null) {
                    try {
                        inputStream.close();
                    } catch (IOException e) {
                        Log.e(TAG, "关闭输入流失败: " + e.getMessage());
                    }
                }
            }
        });

        connectionThreads.put(device.getAddress() + "_receive", receiveThread);
        receiveThread.start();
    }

    /**
     * 发送数据
     */
    public boolean sendData(BluetoothDevice device, byte[] data) {
        BluetoothSocket socket = connectedSockets.get(device.getAddress());
        if (socket == null || !socket.isConnected()) {
            Log.e(TAG, "设备未连接: " + device.getAddress());
            return false;
        }

        try {
            OutputStream outputStream = socket.getOutputStream();
            outputStream.write(data);
            outputStream.flush();
            return true;
        } catch (IOException e) {
            Log.e(TAG, "发送数据失败: " + e.getMessage());
            handleDisconnection(device);
            return false;
        }
    }

    /**
     * 断开连接
     */
    public void disconnectDevice(BluetoothDevice device) {
        String address = device.getAddress();

        // 停止连接线程
        Thread connectThread = connectionThreads.remove(address);
        if (connectThread != null) {
            connectThread.interrupt();
        }

        Thread receiveThread = connectionThreads.remove(address + "_receive");
        if (receiveThread != null) {
            receiveThread.interrupt();
        }

        // 关闭socket
        BluetoothSocket socket = connectedSockets.remove(address);
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException e) {
                Log.e(TAG, "关闭socket失败: " + e.getMessage());
            }
        }

        Log.d(TAG, "设备已断开连接: " + address);

        if (callback != null) {
            callback.onDisconnected(device);
        }
    }

    /**
     * 处理断开连接
     */
    private void handleDisconnection(BluetoothDevice device) {
        connectHandler.post(() -> {
            disconnectDevice(device);
        });
    }

    /**
     * 配对状态变化处理
     */
    private void onBondStateChanged(BluetoothDevice device, Intent intent) {
        int newState = intent.getIntExtra(BluetoothDevice.EXTRA_BOND_STATE, BluetoothDevice.BOND_NONE);
        int prevState = intent.getIntExtra(BluetoothDevice.EXTRA_PREVIOUS_BOND_STATE, BluetoothDevice.BOND_NONE);

        Log.d(TAG, String.format("配对状态变化: %s, %d -> %d",
            device.getAddress(), prevState, newState));

        if (newState == BluetoothDevice.BOND_BONDED) {
            Log.d(TAG, "配对成功: " + device.getAddress());
            if (callback != null) {
                callback.onPairingFinished(device, true);
            }
            // 配对成功后自动连接
            connectDevice(device);
        } else if (newState == BluetoothDevice.BOND_NONE && prevState == BluetoothDevice.BOND_BONDING) {
            Log.d(TAG, "配对失败: " + device.getAddress());
            if (callback != null) {
                callback.onPairingFinished(device, false);
            }
        }
    }

    /**
     * 配对请求处理
     */
    private void onPairingRequest(BluetoothDevice device, Intent intent) {
        int pairingVariant = intent.getIntExtra(BluetoothDevice.EXTRA_PAIRING_VARIANT, -1);
        Log.d(TAG, "配对请求: " + device.getAddress() + ", 类型: " + pairingVariant);

        // 对于某些配对类型，可以自动确认
        if (pairingVariant == BluetoothDevice.PAIRING_VARIANT_PASSKEY_CONFIRMATION) {
            try {
                // 自动确认配对
                Method setPairingConfirmation = device.getClass().getMethod("setPairingConfirmation", boolean.class);
                setPairingConfirmation.invoke(device, true);
            } catch (Exception e) {
                Log.e(TAG, "自动确认配对失败: " + e.getMessage());
            }
        }
    }

    /**
     * 获取连接状态
     */
    public boolean isConnected(BluetoothDevice device) {
        BluetoothSocket socket = connectedSockets.get(device.getAddress());
        return socket != null && socket.isConnected();
    }

    /**
     * 获取已连接的设备列表
     */
    public List<BluetoothDevice> getConnectedDevices() {
        List<BluetoothDevice> devices = new ArrayList<>();
        for (Map.Entry<String, BluetoothSocket> entry : connectedSockets.entrySet()) {
            BluetoothSocket socket = entry.getValue();
            if (socket.isConnected()) {
                // 根据地址获取设备对象
                BluetoothDevice device = bluetoothAdapter.getRemoteDevice(entry.getKey());
                devices.add(device);
            }
        }
        return devices;
    }

    /**
     * 断开所有连接
     */
    public void disconnectAll() {
        for (BluetoothDevice device : getConnectedDevices()) {
            disconnectDevice(device);
        }
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        disconnectAll();

        if (pairingReceiver != null) {
            context.unregisterReceiver(pairingReceiver);
        }

        connectHandler.removeCallbacksAndMessages(null);
    }
}
```

## 26.3 低功耗蓝牙（BLE）开发

低功耗蓝牙特别适用于需要长期运行的小型设备。

### 26.3.1 BLE设备扫描

```java
public class BLEScanner {
    private static final String TAG = "BLEScanner";
    private static final long SCAN_TIMEOUT = 30000; // 30秒扫描超时

    private Context context;
    private BluetoothAdapter bluetoothAdapter;
    private BluetoothLeScanner bleScanner;
    private BLEScanCallback callback;
    private Handler timeoutHandler;
    private boolean isScanning = false;
    private List<ScanResult> scanResults;
    private ScanCallback scanCallback;

    public interface BLEScanCallback {
        void onScanStarted();
        void onScanStopped();
        void onDeviceFound(ScanResult result);
        void onScanError(String error);
    }

    public BLEScanner(Context context, BluetoothAdapter adapter, BLEScanCallback callback) {
        this.context = context;
        this.bluetoothAdapter = adapter;
        this.callback = callback;
        this.scanResults = new ArrayList<>();
        this.timeoutHandler = new Handler(Looper.getMainLooper());

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            this.bleScanner = bluetoothAdapter.getBluetoothLeScanner();
        }

        initializeScanCallback();
    }

    /**
     * 初始化扫描回调
     */
    private void initializeScanCallback() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            scanCallback = new ScanCallback() {
                @Override
                public void onScanResult(int callbackType, ScanResult result) {
                    handleScanResult(result);
                }

                @Override
                public void onBatchScanResults(List<ScanResult> results) {
                    for (ScanResult result : results) {
                        handleScanResult(result);
                    }
                }

                @Override
                public void onScanFailed(int errorCode) {
                    String errorMessage = getScanErrorMessage(errorCode);
                    Log.e(TAG, "BLE扫描失败: " + errorMessage);

                    isScanning = false;
                    timeoutHandler.removeCallbacksAndMessages(null);

                    if (callback != null) {
                        callback.onScanError(errorMessage);
                    }
                }
            };
        }
    }

    /**
     * 开始BLE扫描
     */
    public boolean startScan() {
        if (!isBLESupported()) {
            if (callback != null) {
                callback.onScanError("设备不支持BLE");
            }
            return false;
        }

        if (!hasBluetoothPermissions()) {
            if (callback != null) {
                callback.onScanError("缺少蓝牙权限");
            }
            return false;
        }

        if (isScanning) {
            Log.w(TAG, "已经在扫描中");
            return true;
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && bleScanner != null) {
            scanResults.clear();

            // 配置扫描设置
            ScanSettings settings = new ScanSettings.Builder()
                    .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY) // 高功耗，快速扫描
                    .setCallbackType(ScanSettings.CALLBACK_TYPE_ALL_MATCHES)
                    .setMatchMode(ScanSettings.MATCH_MODE_AGGRESSIVE)
                    .setNumOfMatches(ScanSettings.MATCH_NUM_MAX_ADVERTISEMENT)
                    .build();

            // 配置扫描过滤器
            List<ScanFilter> filters = new ArrayList<>();
            // 可以添加特定设备的过滤条件
            // filters.add(new ScanFilter.Builder().setDeviceName("MyDevice").build());

            try {
                bleScanner.startScan(filters, settings, scanCallback);
                isScanning = true;

                // 设置扫描超时
                timeoutHandler.postDelayed(this::stopScan, SCAN_TIMEOUT);

                Log.d(TAG, "BLE扫描已开始");
                if (callback != null) {
                    callback.onScanStarted();
                }

                return true;
            } catch (Exception e) {
                Log.e(TAG, "开始BLE扫描失败: " + e.getMessage());
                if (callback != null) {
                    callback.onScanError("开始扫描失败: " + e.getMessage());
                }
                return false;
            }
        } else {
            // 使用旧版API
            return startLegacyScan();
        }
    }

    /**
     * 使用旧版API扫描
     */
    @SuppressWarnings("deprecation")
    private boolean startLegacyScan() {
        final LeScanCallback legacyCallback = new LeScanCallback() {
            @Override
            public void onLeScan(BluetoothDevice device, int rssi, byte[] scanRecord) {
                ScanResult result = createScanResult(device, rssi, scanRecord);
                handleScanResult(result);
            }
        };

        boolean started = bluetoothAdapter.startLeScan(legacyCallback);
        if (started) {
            isScanning = true;
            timeoutHandler.postDelayed(() -> {
                bluetoothAdapter.stopLeScan(legacyCallback);
                handleScanStopped();
            }, SCAN_TIMEOUT);

            Log.d(TAG, "Legacy BLE扫描已开始");
            if (callback != null) {
                callback.onScanStarted();
            }
        }

        return started;
    }

    /**
     * 停止BLE扫描
     */
    public void stopScan() {
        if (!isScanning) {
            return;
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && bleScanner != null) {
            bleScanner.stopScan(scanCallback);
        }

        isScanning = false;
        timeoutHandler.removeCallbacksAndMessages(null);
        handleScanStopped();
    }

    /**
     * 处理扫描结果
     */
    private void handleScanResult(ScanResult result) {
        if (result == null || result.getDevice() == null) {
            return;
        }

        BluetoothDevice device = result.getDevice();
        String address = device.getAddress();

        // 检查是否已存在
        for (ScanResult existing : scanResults) {
            if (existing.getDevice().getAddress().equals(address)) {
                // 更新现有结果
                scanResults.remove(existing);
                break;
            }
        }

        scanResults.add(result);

        Log.d(TAG, String.format("发现BLE设备: %s (%s), RSSI: %d dBm",
            device.getName(), address, result.getRssi()));

        if (callback != null) {
            callback.onDeviceFound(result);
        }
    }

    /**
     * 处理扫描停止
     */
    private void handleScanStopped() {
        Log.d(TAG, "BLE扫描已停止，发现设备数: " + scanResults.size());

        if (callback != null) {
            callback.onScanStopped();
        }
    }

    /**
     * 创建ScanResult对象（用于旧版API）
     */
    @TargetApi(Build.VERSION_CODES.LOLLIPOP)
    private ScanResult createScanResult(BluetoothDevice device, int rssi, byte[] scanRecord) {
        ScanRecord record = ScanRecord.parseFromBytes(scanRecord);
        return new ScanResult(device, record, rssi, System.nanoTime());
    }

    /**
     * 获取扫描错误信息
     */
    private String getScanErrorMessage(int errorCode) {
        switch (errorCode) {
            case ScanCallback.SCAN_FAILED_ALREADY_STARTED:
                return "扫描已经在进行中";
            case ScanCallback.SCAN_FAILED_APPLICATION_REGISTRATION_FAILED:
                return "应用注册失败";
            case ScanCallback.SCAN_FAILED_INTERNAL_ERROR:
                return "内部错误";
            case ScanCallback.SCAN_FAILED_FEATURE_UNSUPPORTED:
                return "不支持的功能";
            case ScanCallback.SCAN_FAILED_OUT_OF_HARDWARE_RESOURCES:
                return "硬件资源不足";
            default:
                return "未知错误: " + errorCode;
        }
    }

    /**
     * 检查BLE支持
     */
    public boolean isBLESupported() {
        return bluetoothAdapter != null &&
               context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE);
    }

    /**
     * 检查蓝牙权限
     */
    public boolean hasBluetoothPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED &&
                   ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED;
        } else {
            return ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH) == PackageManager.PERMISSION_GRANTED &&
                   ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_ADMIN) == PackageManager.PERMISSION_GRANTED &&
                   ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        }
    }

    /**
     * 获取扫描结果
     */
    public List<ScanResult> getScanResults() {
        return new ArrayList<>(scanResults);
    }

    /**
     * 按信号强度排序结果
     */
    public void sortResultsByRSSI() {
        scanResults.sort((r1, r2) -> Integer.compare(r2.getRssi(), r1.getRssi()));
    }

    /**
     * 按设备名称过滤结果
     */
    public List<ScanResult> filterResultsByName(String name) {
        List<ScanResult> filtered = new ArrayList<>();
        for (ScanResult result : scanResults) {
            BluetoothDevice device = result.getDevice();
            if (device.getName() != null && device.getName().contains(name)) {
                filtered.add(result);
            }
        }
        return filtered;
    }

    /**
     * 获取设备详细信息
     */
    public String getDeviceScanInfo(ScanResult result) {
        if (result == null || result.getDevice() == null) {
            return "设备信息不可用";
        }

        BluetoothDevice device = result.getDevice();
        ScanRecord scanRecord = result.getScanRecord();

        StringBuilder info = new StringBuilder();
        info.append("设备名称: ").append(device.getName()).append("\n");
        info.append("设备地址: ").append(device.getAddress()).append("\n");
        info.append("信号强度: ").append(result.getRssi()).append(" dBm\n");
        info.append("设备类型: ").append(getDeviceTypeDescription(device.getType())).append("\n");

        if (scanRecord != null) {
            info.append("广播数据长度: ").append(scanRecord.getBytes().length).append(" 字节\n");

            // 获取服务UUID
            List<ParcelUuid> serviceUuids = scanRecord.getServiceUuids();
            if (serviceUuids != null && !serviceUuids.isEmpty()) {
                info.append("服务UUID: ");
                for (ParcelUuid uuid : serviceUuids) {
                    info.append(uuid.toString()).append(" ");
                }
                info.append("\n");
            }

            // 获取制造商数据
            SparseArray<byte[]> manufacturerData = scanRecord.getManufacturerSpecificData();
            if (manufacturerData.size() > 0) {
                info.append("制造商数据: ");
                for (int i = 0; i < manufacturerData.size(); i++) {
                    int manufacturerId = manufacturerData.keyAt(i);
                    byte[] data = manufacturerData.valueAt(i);
                    info.append("ID=").append(manufacturerId).append(", 数据长度=").append(data.length).append(" ");
                }
                info.append("\n");
            }

            // 获取设备外观
            int appearance = scanRecord.getAppearance();
            if (appearance != 0) {
                info.append("设备外观: ").append(appearance).append("\n");
            }

            // 获取发射功率
            int txPower = scanRecord.getTxPowerLevel();
            if (txPower != Integer.MIN_VALUE) {
                info.append("发射功率: ").append(txPower).append(" dBm\n");
            }
        }

        return info.toString();
    }

    private String getDeviceTypeDescription(int type) {
        switch (type) {
            case BluetoothDevice.DEVICE_TYPE_CLASSIC:
                return "经典蓝牙";
            case BluetoothDevice.DEVICE_TYPE_LE:
                return "低功耗蓝牙";
            case BluetoothDevice.DEVICE_TYPE_DUAL:
                return "双模蓝牙";
            default:
                return "未知";
        }
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        stopScan();
        timeoutHandler.removeCallbacksAndMessages(null);
    }
}
```

### 26.3.2 BLE GATT连接和通信

```java
public class BLEGattManager {
    private static final String TAG = "BLEGattManager";
    private static final long CONNECTION_TIMEOUT = 10000; // 10秒连接超时

    private Context context;
    private BluetoothAdapter bluetoothAdapter;
    private GattCallback callback;
    private Handler connectionHandler;
    private Map<String, BluetoothGatt> connectedGatts;
    private Map<String, BluetoothGattCharacteristic> characteristics;

    public interface GattCallback {
        void onConnecting(BluetoothDevice device);
        void onConnected(BluetoothDevice device, BluetoothGatt gatt);
        void onDisconnected(BluetoothDevice device);
        void onServicesDiscovered(BluetoothDevice device, List<BluetoothGattService> services);
        void onCharacteristicRead(BluetoothDevice device, BluetoothGattCharacteristic characteristic);
        void onCharacteristicChanged(BluetoothDevice device, BluetoothGattCharacteristic characteristic);
        void onCharacteristicWrite(BluetoothDevice device, BluetoothGattCharacteristic characteristic);
        void onDescriptorRead(BluetoothDevice device, BluetoothGattDescriptor descriptor);
        void onDescriptorWrite(BluetoothDevice device, BluetoothGattDescriptor descriptor);
        void onReadRemoteRssi(BluetoothDevice device, int rssi);
        void onMtuChanged(BluetoothDevice device, int mtu);
        void onConnectionError(BluetoothDevice device, String error);
    }

    public BLEGattManager(Context context, BluetoothAdapter adapter, GattCallback callback) {
        this.context = context;
        this.bluetoothAdapter = adapter;
        this.callback = callback;
        this.connectedGatts = new HashMap<>();
        this.characteristics = new HashMap<>();
        this.connectionHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * 连接BLE设备
     */
    public boolean connectDevice(BluetoothDevice device, boolean autoConnect) {
        if (device == null) {
            Log.e(TAG, "设备为空");
            return false;
        }

        String address = device.getAddress();

        // 如果已连接，直接返回
        if (connectedGatts.containsKey(address)) {
            Log.d(TAG, "设备已连接: " + address);
            return true;
        }

        if (callback != null) {
            callback.onConnecting(device);
        }

        // 连接设备
        BluetoothGatt gatt;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            gatt = device.connectGatt(context, autoConnect, gattCallback, BluetoothDevice.TRANSPORT_LE);
        } else {
            gatt = device.connectGatt(context, autoConnect, gattCallback);
        }

        if (gatt != null) {
            Log.d(TAG, "开始连接BLE设备: " + address);

            // 设置连接超时
            connectionHandler.postDelayed(() -> {
                if (!connectedGatts.containsKey(address)) {
                    Log.e(TAG, "连接超时: " + address);
                    if (callback != null) {
                        callback.onConnectionError(device, "连接超时");
                    }
                    // 断开连接
                    gatt.disconnect();
                    gatt.close();
                }
            }, CONNECTION_TIMEOUT);

            return true;
        } else {
            Log.e(TAG, "无法创建GATT连接");
            return false;
        }
    }

    /**
     * 断开BLE设备
     */
    public void disconnectDevice(BluetoothDevice device) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.remove(address);

        if (gatt != null) {
            gatt.disconnect();
            gatt.close();
            Log.d(TAG, "设备已断开连接: " + address);
        }
    }

    /**
     * 断开所有设备
     */
    public void disconnectAll() {
        for (Map.Entry<String, BluetoothGatt> entry : connectedGatts.entrySet()) {
            BluetoothGatt gatt = entry.getValue();
            gatt.disconnect();
            gatt.close();
        }
        connectedGatts.clear();
    }

    /**
     * 发现服务
     */
    public boolean discoverServices(BluetoothDevice device) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.get(address);

        if (gatt != null) {
            return gatt.discoverServices();
        }

        Log.e(TAG, "设备未连接: " + address);
        return false;
    }

    /**
     * 读取特征值
     */
    public boolean readCharacteristic(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.get(address);

        if (gatt != null) {
            // 保存特征值引用
            characteristics.put(address + "_" + characteristic.getUuid().toString(), characteristic);
            return gatt.readCharacteristic(characteristic);
        }

        Log.e(TAG, "设备未连接: " + address);
        return false;
    }

    /**
     * 写入特征值
     */
    public boolean writeCharacteristic(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.get(address);

        if (gatt != null) {
            // 保存特征值引用
            characteristics.put(address + "_" + characteristic.getUuid().toString(), characteristic);
            return gatt.writeCharacteristic(characteristic);
        }

        Log.e(TAG, "设备未连接: " + address);
        return false;
    }

    /**
     * 启用特征值通知
     */
    public boolean enableNotification(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.get(address);

        if (gatt != null) {
            // 启用通知
            boolean success = gatt.setCharacteristicNotification(characteristic, true);

            if (success) {
                // 写入描述符以启用通知
                BluetoothGattDescriptor descriptor = characteristic.getDescriptor(
                    UUID.fromString("00002902-0000-1000-8000-00805f9b34fb"));
                if (descriptor != null) {
                    descriptor.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                    success = gatt.writeDescriptor(descriptor);
                }
            }

            if (success) {
                // 保存特征值引用
                characteristics.put(address + "_" + characteristic.getUuid().toString(), characteristic);
            }

            return success;
        }

        Log.e(TAG, "设备未连接: " + address);
        return false;
    }

    /**
     * 禁用特征值通知
     */
    public boolean disableNotification(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.get(address);

        if (gatt != null) {
            // 禁用通知
            boolean success = gatt.setCharacteristicNotification(characteristic, false);

            if (success) {
                // 写入描述符以禁用通知
                BluetoothGattDescriptor descriptor = characteristic.getDescriptor(
                    UUID.fromString("00002902-0000-1000-8000-00805f9b34fb"));
                if (descriptor != null) {
                    descriptor.setValue(BluetoothGattDescriptor.DISABLE_NOTIFICATION_VALUE);
                    success = gatt.writeDescriptor(descriptor);
                }
            }

            return success;
        }

        Log.e(TAG, "设备未连接: " + address);
        return false;
    }

    /**
     * 读取RSSI
     */
    public boolean readRemoteRssi(BluetoothDevice device) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.get(address);

        if (gatt != null) {
            return gatt.readRemoteRssi();
        }

        Log.e(TAG, "设备未连接: " + address);
        return false;
    }

    /**
     * 请求MTU大小
     */
    public boolean requestMtu(BluetoothDevice device, int mtu) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.get(address);

        if (gatt != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            return gatt.requestMtu(mtu);
        }

        Log.e(TAG, "设备未连接或MTU不支持: " + address);
        return false;
    }

    /**
     * GATT回调处理
     */
    private final android.bluetooth.BluetoothGattCallback gattCallback = new android.bluetooth.BluetoothGattCallback() {
        @Override
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            BluetoothDevice device = gatt.getDevice();
            String address = device.getAddress();

            Log.d(TAG, String.format("连接状态变化: %s, status=%d, newState=%d",
                address, status, newState));

            if (newState == BluetoothProfile.STATE_CONNECTED) {
                if (status == BluetoothGatt.GATT_SUCCESS) {
                    // 连接成功
                    connectedGatts.put(address, gatt);

                    connectionHandler.post(() -> {
                        if (callback != null) {
                            callback.onConnected(device, gatt);
                        }
                    });

                    Log.d(TAG, "BLE设备连接成功: " + address);

                    // 自动发现服务
                    gatt.discoverServices();
                } else {
                    // 连接失败
                    Log.e(TAG, "BLE设备连接失败: " + address + ", status: " + status);

                    connectionHandler.post(() -> {
                        if (callback != null) {
                            callback.onConnectionError(device, "连接失败: " + getStatusDescription(status));
                        }
                    });

                    gatt.disconnect();
                    gatt.close();
                }
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                // 断开连接
                connectedGatts.remove(address);

                connectionHandler.post(() -> {
                    if (callback != null) {
                        callback.onDisconnected(device);
                    }
                });

                Log.d(TAG, "BLE设备已断开: " + address);

                // 关闭GATT
                gatt.close();
            }
        }

        @Override
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            BluetoothDevice device = gatt.getDevice();

            if (status == BluetoothGatt.GATT_SUCCESS) {
                List<BluetoothGattService> services = gatt.getServices();
                Log.d(TAG, "发现服务数量: " + services.size());

                connectionHandler.post(() -> {
                    if (callback != null) {
                        callback.onServicesDiscovered(device, services);
                    }
                });

                // 打印服务信息
                for (BluetoothGattService service : services) {
                    Log.d(TAG, "服务: " + service.getUuid());
                    for (BluetoothGattCharacteristic characteristic : service.getCharacteristics()) {
                        Log.d(TAG, "  特征: " + characteristic.getUuid() + ", 属性: " + characteristic.getProperties());
                    }
                }
            } else {
                Log.e(TAG, "服务发现失败: " + status);
            }
        }

        @Override
        public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            BluetoothDevice device = gatt.getDevice();

            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d(TAG, String.format("读取特征值成功: %s = %s",
                    characteristic.getUuid(), bytesToHex(characteristic.getValue())));

                connectionHandler.post(() -> {
                    if (callback != null) {
                        callback.onCharacteristicRead(device, characteristic);
                    }
                });
            } else {
                Log.e(TAG, "读取特征值失败: " + status);
            }
        }

        @Override
        public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic) {
            BluetoothDevice device = gatt.getDevice();

            Log.d(TAG, String.format("特征值通知: %s = %s",
                characteristic.getUuid(), bytesToHex(characteristic.getValue())));

            connectionHandler.post(() -> {
                if (callback != null) {
                    callback.onCharacteristicChanged(device, characteristic);
                }
            });
        }

        @Override
        public void onCharacteristicWrite(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            BluetoothDevice device = gatt.getDevice();

            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d(TAG, "写入特征值成功: " + characteristic.getUuid());

                connectionHandler.post(() -> {
                    if (callback != null) {
                        callback.onCharacteristicWrite(device, characteristic);
                    }
                });
            } else {
                Log.e(TAG, "写入特征值失败: " + status);
            }
        }

        @Override
        public void onDescriptorRead(BluetoothGatt gatt, BluetoothGattDescriptor descriptor, int status) {
            BluetoothDevice device = gatt.getDevice();

            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d(TAG, "读取描述符成功: " + descriptor.getUuid());

                connectionHandler.post(() -> {
                    if (callback != null) {
                        callback.onDescriptorRead(device, descriptor);
                    }
                });
            } else {
                Log.e(TAG, "读取描述符失败: " + status);
            }
        }

        @Override
        public void onDescriptorWrite(BluetoothGatt gatt, BluetoothGattDescriptor descriptor, int status) {
            BluetoothDevice device = gatt.getDevice();

            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d(TAG, "写入描述符成功: " + descriptor.getUuid());

                connectionHandler.post(() -> {
                    if (callback != null) {
                        callback.onDescriptorWrite(device, descriptor);
                    }
                });
            } else {
                Log.e(TAG, "写入描述符失败: " + status);
            }
        }

        @Override
        public void onReadRemoteRssi(BluetoothGatt gatt, int rssi, int status) {
            BluetoothDevice device = gatt.getDevice();

            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d(TAG, "RSSI: " + rssi + " dBm");

                connectionHandler.post(() -> {
                    if (callback != null) {
                        callback.onReadRemoteRssi(device, rssi);
                    }
                });
            } else {
                Log.e(TAG, "读取RSSI失败: " + status);
            }
        }

        @Override
        public void onMtuChanged(BluetoothGatt gatt, int mtu, int status) {
            BluetoothDevice device = gatt.getDevice();

            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d(TAG, "MTU已更改: " + mtu);

                connectionHandler.post(() -> {
                    if (callback != null) {
                        callback.onMtuChanged(device, mtu);
                    }
                });
            } else {
                Log.e(TAG, "MTU更改失败: " + status);
            }
        }
    };

    /**
     * 获取状态描述
     */
    private String getStatusDescription(int status) {
        switch (status) {
            case BluetoothGatt.GATT_SUCCESS:
                return "成功";
            case BluetoothGatt.GATT_FAILURE:
                return "失败";
            case BluetoothGatt.GATT_INSUFFICIENT_AUTHENTICATION:
                return "认证不足";
            case BluetoothGatt.GATT_INSUFFICIENT_ENCRYPTION:
                return "加密不足";
            case BluetoothGatt.GATT_INVALID_OFFSET:
                return "无效偏移";
            case BluetoothGatt.GATT_READ_NOT_PERMITTED:
                return "不允许读取";
            case BluetoothGatt.GATT_REQUEST_NOT_SUPPORTED:
                return "请求不支持";
            case BluetoothGatt.GATT_WRITE_NOT_PERMITTED:
                return "不允许写入";
            default:
                return "未知状态: " + status;
        }
    }

    /**
     * 字节数组转十六进制字符串
     */
    private String bytesToHex(byte[] bytes) {
        if (bytes == null) return "null";

        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02X ", b));
        }
        return sb.toString().trim();
    }

    /**
     * 获取连接状态
     */
    public boolean isConnected(BluetoothDevice device) {
        return connectedGatts.containsKey(device.getAddress());
    }

    /**
     * 获取已连接的设备
     */
    public List<BluetoothDevice> getConnectedDevices() {
        List<BluetoothDevice> devices = new ArrayList<>();
        for (BluetoothGatt gatt : connectedGatts.values()) {
            devices.add(gatt.getDevice());
        }
        return devices;
    }

    /**
     * 获取设备的服务
     */
    public List<BluetoothGattService> getServices(BluetoothDevice device) {
        String address = device.getAddress();
        BluetoothGatt gatt = connectedGatts.get(address);

        if (gatt != null) {
            return gatt.getServices();
        }

        return new ArrayList<>();
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        disconnectAll();
        connectionHandler.removeCallbacksAndMessages(null);
    }
}
```

## 26.4 NFC技术开发

NFC（Near Field Communication）是一种短距离无线通信技术，主要用于移动支付、数据交换和设备配对等场景。

### 26.4.1 NFC基础检测和配置

```java
public class NFCManager {
    private static final String TAG = "NFCManager";

    private Context context;
    private NfcAdapter nfcAdapter;
    private NFCCallback callback;
    private PendingIntent pendingIntent;
    private IntentFilter[] intentFiltersArray;
    private String[][] techListsArray;

    public interface NFCCallback {
        void onNFCDetected(NdefMessage message);
        void onTagDiscovered(Tag tag);
        void onNFCError(String error);
        void onNFCEnabled();
        void onNFCDisabled();
    }

    public NFCManager(Context context, NFCCallback callback) {
        this.context = context;
        this.callback = callback;
        initializeNFC();
        setupForegroundDispatch();
    }

    /**
     * 初始化NFC
     */
    private void initializeNFC() {
        nfcAdapter = NfcAdapter.getDefaultAdapter(context);

        if (nfcAdapter == null) {
            Log.e(TAG, "设备不支持NFC");
            if (callback != null) {
                callback.onNFCError("设备不支持NFC");
            }
        } else {
            Log.d(TAG, "NFC适配器已初始化");
            checkNFCEnabled();
        }
    }

    /**
     * 检查NFC是否启用
     */
    private void checkNFCEnabled() {
        if (nfcAdapter != null) {
            if (nfcAdapter.isEnabled()) {
                Log.d(TAG, "NFC已启用");
                if (callback != null) {
                    callback.onNFCEnabled();
                }
            } else {
                Log.d(TAG, "NFC未启用");
                if (callback != null) {
                    callback.onNFCDisabled();
                }
            }
        }
    }

    /**
     * 设置前台分发
     */
    private void setupForegroundDispatch() {
        if (nfcAdapter == null) return;

        Intent intent = new Intent(context, context.getClass()).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        pendingIntent = PendingIntent.getActivity(context, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        // 设置要拦截的Intent过滤器
        IntentFilter ndef = new IntentFilter(NfcAdapter.ACTION_NDEF_DISCOVERED);
        try {
            ndef.addDataType("*/*");
        } catch (MalformedMimeTypeException e) {
            Log.e(TAG, "设置MIME类型失败: " + e.getMessage());
        }

        IntentFilter tag = new IntentFilter(NfcAdapter.ACTION_TAG_DISCOVERED);
        IntentFilter tech = new IntentFilter(NfcAdapter.ACTION_TECH_DISCOVERED);

        intentFiltersArray = new IntentFilter[]{ndef, tag, tech};

        // 设置要支持的技术列表
        techListsArray = new String[][]{
            new String[]{Ndef.class.getName()},
            new String[]{NdefFormatable.class.getName()},
            new String[]{MifareClassic.class.getName()},
            new String[]{MifareUltralight.class.getName()}
        };
    }

    /**
     * 启用前台分发
     */
    public void enableForegroundDispatch(Activity activity) {
        if (nfcAdapter != null && pendingIntent != null) {
            nfcAdapter.enableForegroundDispatch(activity, pendingIntent, intentFiltersArray, techListsArray);
            Log.d(TAG, "NFC前台分发已启用");
        }
    }

    /**
     * 禁用前台分发
     */
    public void disableForegroundDispatch(Activity activity) {
        if (nfcAdapter != null) {
            nfcAdapter.disableForegroundDispatch(activity);
            Log.d(TAG, "NFC前台分发已禁用");
        }
    }

    /**
     * 处理NFC Intent
     */
    public void handleNFCIntent(Intent intent) {
        if (nfcAdapter == null) return;

        String action = intent.getAction();
        Log.d(TAG, "处理NFC Intent: " + action);

        if (NfcAdapter.ACTION_NDEF_DISCOVERED.equals(action)) {
            handleNDEFDiscovered(intent);
        } else if (NfcAdapter.ACTION_TAG_DISCOVERED.equals(action)) {
            handleTagDiscovered(intent);
        } else if (NfcAdapter.ACTION_TECH_DISCOVERED.equals(action)) {
            handleTechDiscovered(intent);
        }
    }

    /**
     * 处理NDEF发现
     */
    private void handleNDEFDiscovered(Intent intent) {
        NdefMessage[] messages;
        Parcelable[] rawMessages = intent.getParcelableArrayExtra(NfcAdapter.EXTRA_NDEF_MESSAGES);

        if (rawMessages != null) {
            messages = new NdefMessage[rawMessages.length];
            for (int i = 0; i < rawMessages.length; i++) {
                messages[i] = (NdefMessage) rawMessages[i];
            }

            Log.d(TAG, "发现NDEF消息数量: " + messages.length);

            if (messages.length > 0 && callback != null) {
                callback.onNFCDetected(messages[0]);
            }
        } else {
            // 当标签不包含NDEF消息时，尝试从标签中读取
            Tag tag = intent.getParcelableExtra(NfcAdapter.EXTRA_TAG);
            if (tag != null) {
                readNDEFFromTag(tag);
            }
        }
    }

    /**
     * 处理标签发现
     */
    private void handleTagDiscovered(Intent intent) {
        Tag tag = intent.getParcelableExtra(NfcAdapter.EXTRA_TAG);
        if (tag != null) {
            Log.d(TAG, "发现NFC标签: " + bytesToHex(tag.getId()));

            if (callback != null) {
                callback.onTagDiscovered(tag);
            }

            // 尝试读取NDEF消息
            readNDEFFromTag(tag);
        }
    }

    /**
     * 处理技术发现
     */
    private void handleTechDiscovered(Intent intent) {
        Tag tag = intent.getParcelableExtra(NfcAdapter.EXTRA_TAG);
        if (tag != null) {
            Log.d(TAG, "通过技术发现NFC标签");

            if (callback != null) {
                callback.onTagDiscovered(tag);
            }

            readNDEFFromTag(tag);
        }
    }

    /**
     * 从标签读取NDEF消息
     */
    private void readNDEFFromTag(Tag tag) {
        try {
            Ndef ndef = Ndef.get(tag);
            if (ndef != null) {
                ndef.connect();
                NdefMessage message = ndef.getNdefMessage();
                ndef.close();

                if (message != null && callback != null) {
                    callback.onNFCDetected(message);
                }
            } else {
                // 尝试读取NDEF格式化标签
                NdefFormatable formatable = NdefFormatable.get(tag);
                if (formatable != null) {
                    Log.d(TAG, "标签是NDEF格式化的但为空");
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "读取NDEF消息失败: " + e.getMessage());
        }
    }

    /**
     * 写入NDEF消息到标签
     */
    public boolean writeNDEFToTag(Tag tag, NdefMessage message) {
        try {
            Ndef ndef = Ndef.get(tag);
            if (ndef != null) {
                // 标签已包含NDEF数据
                ndef.connect();

                if (!ndef.isWritable()) {
                    Log.e(TAG, "标签不可写");
                    return false;
                }

                int size = message.toByteArray().length;
                if (size > ndef.getMaxSize()) {
                    Log.e(TAG, "消息太大，超出标签容量");
                    return false;
                }

                ndef.writeNdefMessage(message);
                ndef.close();

                Log.d(TAG, "NDEF消息写入成功");
                return true;
            } else {
                // 标签未格式化，尝试格式化并写入
                return formatAndWriteTag(tag, message);
            }
        } catch (Exception e) {
            Log.e(TAG, "写入NDEF消息失败: " + e.getMessage());
            return false;
        }
    }

    /**
     * 格式化并写入标签
     */
    private boolean formatAndWriteTag(Tag tag, NdefMessage message) {
        try {
            NdefFormatable formatable = NdefFormatable.get(tag);
            if (formatable != null) {
                formatable.connect();
                formatable.format(message);
                formatable.close();

                Log.d(TAG, "标签格式化并写入成功");
                return true;
            } else {
                Log.e(TAG, "标签不支持NDEF格式化");
                return false;
            }
        } catch (Exception e) {
            Log.e(TAG, "格式化标签失败: " + e.getMessage());
            return false;
        }
    }

    /**
     * 创建NDEF文本记录
     */
    public static NdefRecord createTextRecord(String text, Locale locale) {
        byte[] langBytes = locale.getLanguage().getBytes(Charset.forName("US-ASCII"));
        byte[] textBytes = text.getBytes(Charset.forName("UTF-8"));

        int langLength = langBytes.length;
        int textLength = textBytes.length;

        byte[] payload = new byte[1 + langLength + textLength];
        payload[0] = (byte) langLength;

        System.arraycopy(langBytes, 0, payload, 1, langLength);
        System.arraycopy(textBytes, 0, payload, 1 + langLength, textLength);

        return new NdefRecord(NdefRecord.TNF_WELL_KNOWN, NdefRecord.RTD_TEXT, new byte[0], payload);
    }

    /**
     * 创建NDEF URI记录
     */
    public static NdefRecord createURIRecord(String uri) {
        return NdefRecord.createUri(uri);
    }

    /**
     * 创建NDEF消息
     */
    public static NdefMessage createNDEFMessage(NdefRecord... records) {
        return new NdefMessage(records);
    }

    /**
     * 解析NDEF文本记录
     */
    public static String parseTextRecord(NdefRecord record) {
        if (record.getTnf() != NdefRecord.TNF_WELL_KNOWN ||
            !Arrays.equals(record.getType(), NdefRecord.RTD_TEXT)) {
            return null;
        }

        byte[] payload = record.getPayload();
        if (payload.length < 1) {
            return null;
        }

        int languageCodeLength = payload[0] & 0x3F;
        byte[] languageCode = new byte[languageCodeLength];
        System.arraycopy(payload, 1, languageCode, 0, languageCodeLength);

        byte[] textBytes = new byte[payload.length - 1 - languageCodeLength];
        System.arraycopy(payload, 1 + languageCodeLength, textBytes, 0, textBytes.length);

        try {
            return new String(textBytes, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            Log.e(TAG, "文本编码错误: " + e.getMessage());
            return null;
        }
    }

    /**
     * 解析NDEF URI记录
     */
    public static String parseURIRecord(NdefRecord record) {
        if (record.getTnf() != NdefRecord.TNF_WELL_KNOWN ||
            !Arrays.equals(record.getType(), NdefRecord.RTD_URI)) {
            return null;
        }

        return record.toUri().toString();
    }

    /**
     * 获取标签详细信息
     */
    public String getTagInfo(Tag tag) {
        if (tag == null) return "标签信息不可用";

        StringBuilder info = new StringBuilder();
        info.append("标签ID: ").append(bytesToHex(tag.getId())).append("\n");

        String[] techList = tag.getTechList();
        info.append("支持的技术:\n");
        for (String tech : techList) {
            info.append("  - ").append(tech).append("\n");
        }

        // 获取标签类型
        if (containsTech(techList, Ndef.class.getName())) {
            info.append("标签类型: NDEF\n");

            try {
                Ndef ndef = Ndef.get(tag);
                if (ndef != null) {
                    info.append("NDEF信息:\n");
                    info.append("  类型: ").append(ndef.getType()).append("\n");
                    info.append("  大小: ").append(ndef.getMaxSize()).append(" 字节\n");
                    info.append("  可写: ").append(ndef.isWritable()).append("\n");

                    NdefMessage message = ndef.getNdefMessage();
                    if (message != null) {
                        info.append("  记录数: ").append(message.getRecords().length).append("\n");
                    }
                }
            } catch (Exception e) {
                info.append("  获取NDEF信息失败: ").append(e.getMessage()).append("\n");
            }
        }

        if (containsTech(techList, MifareClassic.class.getName())) {
            info.append("标签类型: Mifare Classic\n");
            try {
                MifareClassic mifare = MifareClassic.get(tag);
                if (mifare != null) {
                    info.append("  扇区数: ").append(mifare.getSectorCount()).append("\n");
                    info.append("  块数: ").append(mifare.getBlockCount()).append("\n");
                    info.append("  大小: ").append(mifare.getSize()).append(" 字节\n");
                }
            } catch (Exception e) {
                info.append("  获取Mifare信息失败: ").append(e.getMessage()).append("\n");
            }
        }

        return info.toString();
    }

    /**
     * 检查是否包含特定技术
     */
    private boolean containsTech(String[] techList, String tech) {
        for (String t : techList) {
            if (tech.equals(t)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 字节数组转十六进制字符串
     */
    private String bytesToHex(byte[] bytes) {
        if (bytes == null) return "null";

        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02X", b));
        }
        return sb.toString();
    }

    /**
     * 检查NFC支持
     */
    public boolean isNFCSupported() {
        return nfcAdapter != null;
    }

    /**
     * 检查NFC是否启用
     */
    public boolean isNFCEnabled() {
        return nfcAdapter != null && nfcAdapter.isEnabled();
    }

    /**
     * 启用NFC设置界面
     */
    public void enableNFCSettings(Activity activity) {
        if (nfcAdapter != null && !nfcAdapter.isEnabled()) {
            Intent intent = new Intent(Settings.ACTION_NFC_SETTINGS);
            activity.startActivity(intent);
        }
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        pendingIntent = null;
        intentFiltersArray = null;
        techListsArray = null;
    }
}
```

## 26.5 综合应用：智能设备管理器

结合蓝牙和NFC技术，创建一个智能设备管理器：

```java
public class SmartDeviceManager {
    private static final String TAG = "SmartDeviceManager";

    private Context context;
    private BluetoothAdapterManager bluetoothManager;
    private BluetoothDeviceDiscovery deviceDiscovery;
    private BluetoothConnectionManager connectionManager;
    private BLEScanner bleScanner;
    private BLEGattManager gattManager;
    private NFCManager nfcManager;

    private DeviceManagementCallback callback;
    private Map<String, SmartDevice> managedDevices;
    private List<SmartDevice> discoveredDevices;

    public interface DeviceManagementCallback {
        void onDeviceDiscovered(SmartDevice device);
        void onDeviceConnected(SmartDevice device);
        void onDeviceDisconnected(SmartDevice device);
        void onDeviceDataReceived(SmartDevice device, byte[] data);
        void onNFCTagDetected(NFCInfo nfcInfo);
        void onManagementError(String error);
    }

    public SmartDeviceManager(Context context, DeviceManagementCallback callback) {
        this.context = context;
        this.callback = callback;
        this.managedDevices = new HashMap<>();
        this.discoveredDevices = new ArrayList<>();

        initializeManagers();
    }

    /**
     * 初始化管理器
     */
    private void initializeManagers() {
        // 蓝牙管理器
        bluetoothManager = new BluetoothAdapterManager(context, new BluetoothAdapterManager.BluetoothCallback() {
            @Override
            public void onBluetoothEnabled() {
                Log.d(TAG, "蓝牙已启用");
                startDeviceDiscovery();
            }

            @Override
            public void onBluetoothDisabled() {
                Log.d(TAG, "蓝牙已禁用");
            }

            @Override
            public void onBluetoothError(String error) {
                Log.e(TAG, "蓝牙错误: " + error);
                if (callback != null) {
                    callback.onManagementError("蓝牙错误: " + error);
                }
            }

            @Override
            public void onPermissionDenied() {
                Log.e(TAG, "蓝牙权限被拒绝");
                if (callback != null) {
                    callback.onManagementError("蓝牙权限被拒绝");
                }
            }
        });

        // 设备发现
        deviceDiscovery = new BluetoothDeviceDiscovery(context,
            bluetoothManager.getBluetoothAdapter(), new BluetoothDeviceDiscovery.DeviceDiscoveryCallback() {
            @Override
            public void onDeviceDiscovered(BluetoothDevice device, int rssi, byte[] scanRecord) {
                handleBluetoothDeviceDiscovered(device, rssi, scanRecord);
            }

            @Override
            public void onDiscoveryStarted() {
                Log.d(TAG, "蓝牙设备发现已开始");
            }

            @Override
            public void onDiscoveryFinished() {
                Log.d(TAG, "蓝牙设备发现已完成");
            }

            @Override
            public void onDiscoveryError(String error) {
                Log.e(TAG, "设备发现错误: " + error);
                if (callback != null) {
                    callback.onManagementError(error);
                }
            }

            @Override
            public void onBluetoothStateChanged(int state) {
                Log.d(TAG, "蓝牙状态变化: " + state);
            }
        });

        // 连接管理器
        connectionManager = new BluetoothConnectionManager(context,
            bluetoothManager.getBluetoothAdapter(), new BluetoothConnectionManager.ConnectionCallback() {
            @Override
            public void onPairingStarted(BluetoothDevice device) {
                Log.d(TAG, "开始配对: " + device.getAddress());
            }

            @Override
            public void onPairingFinished(BluetoothDevice device, boolean success) {
                Log.d(TAG, "配对完成: " + device.getAddress() + ", 成功: " + success);
            }

            @Override
            public void onConnecting(BluetoothDevice device) {
                Log.d(TAG, "正在连接: " + device.getAddress());
            }

            @Override
            public void onConnected(BluetoothDevice device, BluetoothSocket socket) {
                handleBluetoothDeviceConnected(device, socket);
            }

            @Override
            public void onDisconnected(BluetoothDevice device) {
                handleBluetoothDeviceDisconnected(device);
            }

            @Override
            public void onConnectionError(BluetoothDevice device, String error) {
                Log.e(TAG, "连接错误: " + device.getAddress() + ", " + error);
                if (callback != null) {
                    callback.onManagementError(error);
                }
            }

            @Override
            public void onDataReceived(BluetoothDevice device, byte[] data) {
                handleBluetoothDataReceived(device, data);
            }
        });

        // BLE扫描器
        bleScanner = new BLEScanner(context, bluetoothManager.getBluetoothAdapter(), new BLEScanner.BLEScanCallback() {
            @Override
            public void onScanStarted() {
                Log.d(TAG, "BLE扫描已开始");
            }

            @Override
            public void onScanStopped() {
                Log.d(TAG, "BLE扫描已停止");
            }

            @Override
            public void onDeviceFound(ScanResult result) {
                handleBLEDeviceDiscovered(result);
            }

            @Override
            public void onScanError(String error) {
                Log.e(TAG, "BLE扫描错误: " + error);
                if (callback != null) {
                    callback.onManagementError(error);
                }
            }
        });

        // BLE GATT管理器
        gattManager = new BLEGattManager(context, bluetoothManager.getBluetoothAdapter(), new BLEGattManager.GattCallback() {
            @Override
            public void onConnecting(BluetoothDevice device) {
                Log.d(TAG, "BLE正在连接: " + device.getAddress());
            }

            @Override
            public void onConnected(BluetoothDevice device, BluetoothGatt gatt) {
                handleBLEDeviceConnected(device, gatt);
            }

            @Override
            public void onDisconnected(BluetoothDevice device) {
                handleBLEDeviceDisconnected(device);
            }

            @Override
            public void onServicesDiscovered(BluetoothDevice device, List<BluetoothGattService> services) {
                handleBLEServicesDiscovered(device, services);
            }

            @Override
            public void onCharacteristicRead(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
                handleBLECharacteristicRead(device, characteristic);
            }

            @Override
            public void onCharacteristicChanged(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
                handleBLECharacteristicChanged(device, characteristic);
            }

            @Override
            public void onCharacteristicWrite(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
                Log.d(TAG, "BLE特征值写入: " + device.getAddress());
            }

            @Override
            public void onDescriptorRead(BluetoothDevice device, BluetoothGattDescriptor descriptor) {
                Log.d(TAG, "BLE描述符读取: " + device.getAddress());
            }

            @Override
            public void onDescriptorWrite(BluetoothDevice device, BluetoothGattDescriptor descriptor) {
                Log.d(TAG, "BLE描述符写入: " + device.getAddress());
            }

            @Override
            public void onReadRemoteRssi(BluetoothDevice device, int rssi) {
                Log.d(TAG, "BLE RSSI: " + device.getAddress() + " = " + rssi + " dBm");
            }

            @Override
            public void onMtuChanged(BluetoothDevice device, int mtu) {
                Log.d(TAG, "BLE MTU变更: " + device.getAddress() + " = " + mtu);
            }

            @Override
            public void onConnectionError(BluetoothDevice device, String error) {
                Log.e(TAG, "BLE连接错误: " + device.getAddress() + ", " + error);
                if (callback != null) {
                    callback.onManagementError(error);
                }
            }
        });

        // NFC管理器
        nfcManager = new NFCManager(context, new NFCManager.NFCCallback() {
            @Override
            public void onNFCDetected(NdefMessage message) {
                handleNFCMessage(message);
            }

            @Override
            public void onTagDiscovered(Tag tag) {
                handleNFCTag(tag);
            }

            @Override
            public void onNFCError(String error) {
                Log.e(TAG, "NFC错误: " + error);
                if (callback != null) {
                    callback.onManagementError(error);
                }
            }

            @Override
            public void onNFCEnabled() {
                Log.d(TAG, "NFC已启用");
            }

            @Override
            public void onNFCDisabled() {
                Log.d(TAG, "NFC未启用");
            }
        });
    }

    /**
     * 开始设备发现
     */
    public void startDeviceDiscovery() {
        if (!bluetoothManager.isBluetoothEnabled()) {
            Log.w(TAG, "蓝牙未启用，无法开始设备发现");
            return;
        }

        // 清理之前的发现结果
        discoveredDevices.clear();

        // 开始传统蓝牙设备发现
        deviceDiscovery.startDiscovery();

        // 开始BLE设备扫描
        if (bleScanner.isBLESupported()) {
            bleScanner.startScan();
        }
    }

    /**
     * 停止设备发现
     */
    public void stopDeviceDiscovery() {
        deviceDiscovery.stopDiscovery();
        bleScanner.stopScan();
    }

    /**
     * 处理蓝牙设备发现
     */
    private void handleBluetoothDeviceDiscovered(BluetoothDevice device, int rssi, byte[] scanRecord) {
        SmartDevice smartDevice = createSmartDevice(device, rssi, SmartDevice.DeviceType.CLASSIC_BLUETOOTH);
        addDiscoveredDevice(smartDevice);
    }

    /**
     * 处理BLE设备发现
     */
    private void handleBLEDeviceDiscovered(ScanResult result) {
        BluetoothDevice device = result.getDevice();
        SmartDevice smartDevice = createSmartDevice(device, result.getRssi(), SmartDevice.DeviceType.BLE);
        smartDevice.setScanResult(result);
        addDiscoveredDevice(smartDevice);
    }

    /**
     * 创建智能设备对象
     */
    private SmartDevice createSmartDevice(BluetoothDevice device, int rssi, SmartDevice.DeviceType type) {
        SmartDevice smartDevice = new SmartDevice();
        smartDevice.setBluetoothDevice(device);
        smartDevice.setAddress(device.getAddress());
        smartDevice.setName(device.getName() != null ? device.getName() : "未知设备");
        smartDevice.setRssi(rssi);
        smartDevice.setDeviceType(type);
        smartDevice.setDiscoveryTime(System.currentTimeMillis());

        return smartDevice;
    }

    /**
     * 添加发现的设备
     */
    private void addDiscoveredDevice(SmartDevice device) {
        // 检查是否已存在
        for (SmartDevice existing : discoveredDevices) {
            if (existing.getAddress().equals(device.getAddress())) {
                // 更新现有设备信息
                existing.setRssi(device.getRssi());
                existing.setDiscoveryTime(System.currentTimeMillis());
                if (device.getScanResult() != null) {
                    existing.setScanResult(device.getScanResult());
                }
                return;
            }
        }

        discoveredDevices.add(device);

        Log.d(TAG, "发现智能设备: " + device.getName() + " (" + device.getAddress() + ")");

        if (callback != null) {
            callback.onDeviceDiscovered(device);
        }
    }

    /**
     * 连接设备
     */
    public void connectDevice(SmartDevice device) {
        if (device.getDeviceType() == SmartDevice.DeviceType.BLE) {
            // 连接BLE设备
            gattManager.connectDevice(device.getBluetoothDevice(), false);
        } else {
            // 连接传统蓝牙设备
            connectionManager.connectDevice(device.getBluetoothDevice());
        }

        device.setConnectionStatus(SmartDevice.ConnectionStatus.CONNECTING);
        managedDevices.put(device.getAddress(), device);
    }

    /**
     * 断开设备连接
     */
    public void disconnectDevice(SmartDevice device) {
        if (device.getDeviceType() == SmartDevice.DeviceType.BLE) {
            gattManager.disconnectDevice(device.getBluetoothDevice());
        } else {
            connectionManager.disconnectDevice(device.getBluetoothDevice());
        }

        device.setConnectionStatus(SmartDevice.ConnectionStatus.DISCONNECTED);
        managedDevices.remove(device.getAddress());
    }

    /**
     * 处理蓝牙设备连接
     */
    private void handleBluetoothDeviceConnected(BluetoothDevice device, BluetoothSocket socket) {
        SmartDevice smartDevice = managedDevices.get(device.getAddress());
        if (smartDevice != null) {
            smartDevice.setConnectionStatus(SmartDevice.ConnectionStatus.CONNECTED);
            smartDevice.setBluetoothSocket(socket);

            Log.d(TAG, "蓝牙设备已连接: " + device.getAddress());

            if (callback != null) {
                callback.onDeviceConnected(smartDevice);
            }
        }
    }

    /**
     * 处理BLE设备连接
     */
    private void handleBLEDeviceConnected(BluetoothDevice device, BluetoothGatt gatt) {
        SmartDevice smartDevice = managedDevices.get(device.getAddress());
        if (smartDevice != null) {
            smartDevice.setConnectionStatus(SmartDevice.ConnectionStatus.CONNECTED);
            smartDevice.setBluetoothGatt(gatt);

            Log.d(TAG, "BLE设备已连接: " + device.getAddress());

            if (callback != null) {
                callback.onDeviceConnected(smartDevice);
            }
        }
    }

    /**
     * 处理蓝牙设备断开连接
     */
    private void handleBluetoothDeviceDisconnected(BluetoothDevice device) {
        SmartDevice smartDevice = managedDevices.remove(device.getAddress());
        if (smartDevice != null) {
            smartDevice.setConnectionStatus(SmartDevice.ConnectionStatus.DISCONNECTED);

            Log.d(TAG, "蓝牙设备已断开: " + device.getAddress());

            if (callback != null) {
                callback.onDeviceDisconnected(smartDevice);
            }
        }
    }

    /**
     * 处理BLE设备断开连接
     */
    private void handleBLEDeviceDisconnected(BluetoothDevice device) {
        SmartDevice smartDevice = managedDevices.remove(device.getAddress());
        if (smartDevice != null) {
            smartDevice.setConnectionStatus(SmartDevice.ConnectionStatus.DISCONNECTED);

            Log.d(TAG, "BLE设备已断开: " + device.getAddress());

            if (callback != null) {
                callback.onDeviceDisconnected(smartDevice);
            }
        }
    }

    /**
     * 处理BLE服务发现
     */
    private void handleBLEServicesDiscovered(BluetoothDevice device, List<BluetoothGattService> services) {
        SmartDevice smartDevice = managedDevices.get(device.getAddress());
        if (smartDevice != null) {
            smartDevice.setGattServices(services);
            Log.d(TAG, "BLE服务发现完成: " + device.getAddress() + ", 服务数: " + services.size());

            // 可以根据服务类型识别设备
            identifyDeviceByServices(smartDevice, services);
        }
    }

    /**
     * 根据服务识别设备
     */
    private void identifyDeviceByServices(SmartDevice device, List<BluetoothGattService> services) {
        for (BluetoothGattService service : services) {
            UUID serviceUuid = service.getUuid();

            // 心率服务
            if (serviceUuid.toString().startsWith("0000180d")) {
                device.setDeviceCategory(SmartDevice.DeviceCategory.HEALTH_MONITOR);
                device.setDeviceFunction("心率监测");
            }
            // 电池服务
            else if (serviceUuid.toString().startsWith("0000180f")) {
                device.setHasBattery(true);
            }
            // 环境传感服务
            else if (serviceUuid.toString().startsWith("0000181a")) {
                device.setDeviceCategory(SmartDevice.DeviceCategory.SENSOR);
                device.setDeviceFunction("环境监测");
            }
            // 通用用户界面服务
            else if (serviceUuid.toString().startsWith("0000181c")) {
                device.setDeviceCategory(SmartDevice.DeviceCategory.CONTROLLER);
                device.setDeviceFunction("控制器");
            }
        }
    }

    /**
     * 处理蓝牙数据接收
     */
    private void handleBluetoothDataReceived(BluetoothDevice device, byte[] data) {
        SmartDevice smartDevice = managedDevices.get(device.getAddress());
        if (smartDevice != null) {
            Log.d(TAG, "接收到蓝牙数据: " + device.getAddress() + ", 长度: " + data.length);

            if (callback != null) {
                callback.onDeviceDataReceived(smartDevice, data);
            }
        }
    }

    /**
     * 处理BLE特征值读取
     */
    private void handleBLECharacteristicRead(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
        SmartDevice smartDevice = managedDevices.get(device.getAddress());
        if (smartDevice != null) {
            Log.d(TAG, "BLE特征值读取: " + device.getAddress() + ", " + characteristic.getUuid());

            if (callback != null) {
                callback.onDeviceDataReceived(smartDevice, characteristic.getValue());
            }
        }
    }

    /**
     * 处理BLE特征值变化
     */
    private void handleBLECharacteristicChanged(BluetoothDevice device, BluetoothGattCharacteristic characteristic) {
        SmartDevice smartDevice = managedDevices.get(device.getAddress());
        if (smartDevice != null) {
            Log.d(TAG, "BLE特征值变化: " + device.getAddress() + ", " + characteristic.getUuid());

            if (callback != null) {
                callback.onDeviceDataReceived(smartDevice, characteristic.getValue());
            }
        }
    }

    /**
     * 处理NFC消息
     */
    private void handleNFCMessage(NdefMessage message) {
        NFCInfo nfcInfo = parseNDEFMessage(message);

        Log.d(TAG, "检测到NFC消息: " + nfcInfo.toString());

        if (callback != null) {
            callback.onNFCTagDetected(nfcInfo);
        }

        // 尝试通过NFC信息连接蓝牙设备
        connectDeviceViaNFC(nfcInfo);
    }

    /**
     * 处理NFC标签
     */
    private void handleNFCTag(Tag tag) {
        String tagInfo = nfcManager.getTagInfo(tag);
        Log.d(TAG, "检测到NFC标签: " + tagInfo);

        NFCInfo nfcInfo = new NFCInfo();
        nfcInfo.setTagId(bytesToHex(tag.getId()));
        nfcInfo.setTagInfo(tagInfo);

        if (callback != null) {
            callback.onNFCTagDetected(nfcInfo);
        }
    }

    /**
     * 解析NDEF消息
     */
    private NFCInfo parseNDEFMessage(NdefMessage message) {
        NFCInfo nfcInfo = new NFCInfo();

        NdefRecord[] records = message.getRecords();
        for (NdefRecord record : records) {
            if (record.getTnf() == NdefRecord.TNF_WELL_KNOWN) {
                if (Arrays.equals(record.getType(), NdefRecord.RTD_TEXT)) {
                    nfcInfo.setText(NFCManager.parseTextRecord(record));
                } else if (Arrays.equals(record.getType(), NdefRecord.RTD_URI)) {
                    nfcInfo.setUri(NFCManager.parseURIRecord(record));
                }
            }

            // 尝试解析蓝牙设备地址
            parseBluetoothAddressFromRecord(record, nfcInfo);
        }

        return nfcInfo;
    }

    /**
     * 从NDEF记录解析蓝牙地址
     */
    private void parseBluetoothAddressFromRecord(NdefRecord record, NFCInfo nfcInfo) {
        byte[] payload = record.getPayload();
        if (payload != null && payload.length >= 6) {
            // 检查是否是蓝牙地址格式
            String address = String.format("%02X:%02X:%02X:%02X:%02X:%02X",
                payload[0], payload[1], payload[2], payload[3], payload[4], payload[5]);

            if (isValidBluetoothAddress(address)) {
                nfcInfo.setBluetoothAddress(address);
            }
        }
    }

    /**
     * 验证蓝牙地址格式
     */
    private boolean isValidBluetoothAddress(String address) {
        return address != null && address.matches("^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$");
    }

    /**
     * 通过NFC连接设备
     */
    private void connectDeviceViaNFC(NFCInfo nfcInfo) {
        String bluetoothAddress = nfcInfo.getBluetoothAddress();
        if (bluetoothAddress != null && isValidBluetoothAddress(bluetoothAddress)) {
            // 查找已发现的设备
            for (SmartDevice device : discoveredDevices) {
                if (device.getAddress().equals(bluetoothAddress)) {
                    Log.d(TAG, "通过NFC连接设备: " + bluetoothAddress);
                    connectDevice(device);
                    return;
                }
            }

            // 如果未在已发现设备中找到，尝试直接连接
            BluetoothAdapter adapter = bluetoothManager.getBluetoothAdapter();
            if (adapter != null) {
                BluetoothDevice device = adapter.getRemoteDevice(bluetoothAddress);
                SmartDevice smartDevice = createSmartDevice(device, 0, SmartDevice.DeviceType.CLASSIC_BLUETOOTH);
                connectDevice(smartDevice);
            }
        }
    }

    /**
     * 发送数据到设备
     */
    public boolean sendDataToDevice(SmartDevice device, byte[] data) {
        if (device.getDeviceType() == SmartDevice.DeviceType.BLE) {
            // BLE数据发送需要通过特征值
            return false; // 需要特定的特征值
        } else {
            // 传统蓝牙数据发送
            return connectionManager.sendData(device.getBluetoothDevice(), data);
        }
    }

    /**
     * 获取已发现的设备列表
     */
    public List<SmartDevice> getDiscoveredDevices() {
        return new ArrayList<>(discoveredDevices);
    }

    /**
     * 获取已连接的设备列表
     */
    public List<SmartDevice> getConnectedDevices() {
        List<SmartDevice> connected = new ArrayList<>();
        for (SmartDevice device : managedDevices.values()) {
            if (device.getConnectionStatus() == SmartDevice.ConnectionStatus.CONNECTED) {
                connected.add(device);
            }
        }
        return connected;
    }

    /**
     * 获取设备详细信息
     */
    public String getDeviceInfo(SmartDevice device) {
        StringBuilder info = new StringBuilder();
        info.append("设备名称: ").append(device.getName()).append("\n");
        info.append("设备地址: ").append(device.getAddress()).append("\n");
        info.append("设备类型: ").append(device.getDeviceType().toString()).append("\n");
        info.append("设备类别: ").append(device.getDeviceCategory().toString()).append("\n");
        info.append("设备功能: ").append(device.getDeviceFunction()).append("\n");
        info.append("连接状态: ").append(device.getConnectionStatus().toString()).append("\n");
        info.append("信号强度: ").append(device.getRssi()).append(" dBm\n");
        info.append("发现时间: ").append(new Date(device.getDiscoveryTime()).toString()).append("\n");

        if (device.getBluetoothDevice() != null) {
            info.append("\n蓝牙详细信息:\n");
            info.append(deviceDiscovery.getDeviceInfo(device.getBluetoothDevice()));
        }

        if (device.getScanResult() != null) {
            info.append("\nBLE扫描信息:\n");
            info.append(bleScanner.getDeviceScanInfo(device.getScanResult()));
        }

        if (device.getGattServices() != null && !device.getGattServices().isEmpty()) {
            info.append("\nGATT服务:\n");
            for (BluetoothGattService service : device.getGattServices()) {
                info.append("  服务: ").append(service.getUuid()).append("\n");
                for (BluetoothGattCharacteristic characteristic : service.getCharacteristics()) {
                    info.append("    特征: ").append(characteristic.getUuid()).append("\n");
                }
            }
        }

        return info.toString();
    }

    /**
     * 字节数组转十六进制字符串
     */
    private String bytesToHex(byte[] bytes) {
        if (bytes == null) return "null";

        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02X", b));
        }
        return sb.toString();
    }

    /**
     * 清理资源
     */
    public void cleanup() {
        stopDeviceDiscovery();
        connectionManager.cleanup();
        gattManager.cleanup();
        bleScanner.cleanup();
        deviceDiscovery.cleanup();
        nfcManager.cleanup();

        managedDevices.clear();
        discoveredDevices.clear();
    }

    /**
     * 智能设备模型类
     */
    public static class SmartDevice {
        public enum DeviceType {
            CLASSIC_BLUETOOTH, BLE, DUAL_MODE
        }

        public enum ConnectionStatus {
            DISCONNECTED, CONNECTING, CONNECTED
        }

        public enum DeviceCategory {
            UNKNOWN, AUDIO_VIDEO, COMPUTER, HEALTH_MONITOR, SENSOR,
            CONTROLLER, PHONE, WEARABLE, TOY
        }

        private BluetoothDevice bluetoothDevice;
        private BluetoothSocket bluetoothSocket;
        private BluetoothGatt bluetoothGatt;
        private ScanResult scanResult;
        private List<BluetoothGattService> gattServices;

        private String address;
        private String name;
        private String deviceFunction;
        private DeviceType deviceType;
        private DeviceCategory deviceCategory;
        private ConnectionStatus connectionStatus;
        private int rssi;
        private long discoveryTime;
        private boolean hasBattery;

        public SmartDevice() {
            this.deviceType = DeviceType.CLASSIC_BLUETOOTH;
            this.deviceCategory = DeviceCategory.UNKNOWN;
            this.connectionStatus = ConnectionStatus.DISCONNECTED;
            this.deviceFunction = "未知";
            this.hasBattery = false;
        }

        // Getters and Setters
        public BluetoothDevice getBluetoothDevice() { return bluetoothDevice; }
        public void setBluetoothDevice(BluetoothDevice bluetoothDevice) { this.bluetoothDevice = bluetoothDevice; }

        public BluetoothSocket getBluetoothSocket() { return bluetoothSocket; }
        public void setBluetoothSocket(BluetoothSocket bluetoothSocket) { this.bluetoothSocket = bluetoothSocket; }

        public BluetoothGatt getBluetoothGatt() { return bluetoothGatt; }
        public void setBluetoothGatt(BluetoothGatt bluetoothGatt) { this.bluetoothGatt = bluetoothGatt; }

        public ScanResult getScanResult() { return scanResult; }
        public void setScanResult(ScanResult scanResult) { this.scanResult = scanResult; }

        public List<BluetoothGattService> getGattServices() { return gattServices; }
        public void setGattServices(List<BluetoothGattService> gattServices) { this.gattServices = gattServices; }

        public String getAddress() { return address; }
        public void setAddress(String address) { this.address = address; }

        public String getName() { return name; }
        public void setName(String name) { this.name = name; }

        public String getDeviceFunction() { return deviceFunction; }
        public void setDeviceFunction(String deviceFunction) { this.deviceFunction = deviceFunction; }

        public DeviceType getDeviceType() { return deviceType; }
        public void setDeviceType(DeviceType deviceType) { this.deviceType = deviceType; }

        public DeviceCategory getDeviceCategory() { return deviceCategory; }
        public void setDeviceCategory(DeviceCategory deviceCategory) { this.deviceCategory = deviceCategory; }

        public ConnectionStatus getConnectionStatus() { return connectionStatus; }
        public void setConnectionStatus(ConnectionStatus connectionStatus) { this.connectionStatus = connectionStatus; }

        public int getRssi() { return rssi; }
        public void setRssi(int rssi) { this.rssi = rssi; }

        public long getDiscoveryTime() { return discoveryTime; }
        public void setDiscoveryTime(long discoveryTime) { this.discoveryTime = discoveryTime; }

        public boolean hasBattery() { return hasBattery; }
        public void setHasBattery(boolean hasBattery) { this.hasBattery = hasBattery; }
    }

    /**
     * NFC信息类
     */
    public static class NFCInfo {
        private String tagId;
        private String tagInfo;
        private String text;
        private String uri;
        private String bluetoothAddress;

        // Getters and Setters
        public String getTagId() { return tagId; }
        public void setTagId(String tagId) { this.tagId = tagId; }

        public String getTagInfo() { return tagInfo; }
        public void setTagInfo(String tagInfo) { this.tagInfo = tagInfo; }

        public String getText() { return text; }
        public void setText(String text) { this.text = text; }

        public String getUri() { return uri; }
        public void setUri(String uri) { this.uri = uri; }

        public String getBluetoothAddress() { return bluetoothAddress; }
        public void setBluetoothAddress(String bluetoothAddress) { this.bluetoothAddress = bluetoothAddress; }

        @Override
        public String toString() {
            return "NFCInfo{" +
                   "tagId='" + tagId + '\'' +
                   ", text='" + text + '\'' +
                   ", uri='" + uri + '\'' +
                   ", bluetoothAddress='" + bluetoothAddress + '\'' +
                   '}';
        }
    }
}
```

## 26.6 总结

本章详细介绍了Android蓝牙和NFC近场通信的开发技术，包括：

### 26.6.1 主要内容回顾

1. **蓝牙技术基础**
   - 蓝牙技术分类和特点
   - 蓝牙适配器管理
   - 权限管理和状态检测

2. **经典蓝牙开发**
   - 设备发现和扫描
   - 设备配对和连接
   - 数据传输和通信
   - 连接管理和状态监控

3. **低功耗蓝牙（BLE）开发**
   - BLE设备扫描和过滤
   - GATT服务和特征值操作
   - 连接管理和MTU协商
   - 通知和指示功能

4. **NFC技术开发**
   - NFC标签检测和读写
   - NDEF消息解析和创建
   - 前台分发系统
   - NFC与蓝牙的集成应用

5. **综合应用案例**
   - 智能设备管理器的设计
   - 多协议设备发现和连接
   - 统一的设备管理接口
   - NFC快速配对功能

### 26.6.2 最佳实践总结

1. **权限管理**
   - 动态请求蓝牙和位置权限
   - 提供清晰的权限使用说明
   - 处理权限拒绝的情况

2. **连接管理**
   - 实现可靠的连接超时机制
   - 正确处理连接状态变化
   - 及时释放连接资源

3. **性能优化**
   - 合理设置扫描间隔和超时
   - 避免频繁的设备发现操作
   - 使用后台线程处理数据传输

4. **用户体验**
   - 提供清晰的连接状态反馈
   - 实现友好的错误处理机制
   - 支持设备信息的详细显示

5. **安全考虑**
   - 验证设备身份和合法性
   - 加密敏感数据传输
   - 实现安全的配对机制

### 26.6.3 下一步学习

掌握了蓝牙和NFC技术后，读者可以继续学习：
- Android Things和IoT设备开发
- 可穿戴设备应用开发
- 车载系统应用开发
- 智能家居集成技术
- WebRTC实时通信技术

通过本章的学习，读者应该能够熟练使用Android的蓝牙和NFC API，开发出功能丰富的近场通信应用。下一章将开始第七部分：现代Android开发的内容。