import os, re

base = '/data/workspace/dehaze-system/dehaze-react-native'

# --- Fix ImmersiveHeader.tsx: remove unused 'colors' import ---
fp = os.path.join(base, 'src/layout/components/ImmersiveHeader.tsx')
with open(fp) as f: c = f.read()
c = c.replace("import { colors } from '@/theme/colors';\n", '')
with open(fp, 'w') as f: f.write(c)
print('FIXED: ImmersiveHeader.tsx - removed colors import')

# --- Fix batch/index.tsx: remove unused 'Badge' import ---
fp = os.path.join(base, 'src/pages/batch/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("import Badge from '@/components/Badge';\n", '')
with open(fp, 'w') as f: f.write(c)
print('FIXED: batch/index.tsx - removed Badge import')

# --- Fix dashboard/index.tsx: remove unused 'FeedbackAPI' ---
fp = os.path.join(base, 'src/pages/dashboard/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("import { AlgorithmAPI, DatasetAPI, TaskAPI, OrderAPI, FeedbackAPI } from 'dehaze-sdk-js';",
               "import { AlgorithmAPI, DatasetAPI, TaskAPI, OrderAPI } from 'dehaze-sdk-js';")
with open(fp, 'w') as f: f.write(c)
print('FIXED: dashboard/index.tsx - removed FeedbackAPI')

# --- Fix notify/index.tsx: remove TouchableOpacity, toggleModuleSwitch, moduleSwitches ---
fp = os.path.join(base, 'src/pages/notify/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("import {\n  View,\n  Text,\n  ScrollView,\n  StyleSheet,\n  Switch,\n  TouchableOpacity,\n  Alert,\n} from 'react-native';",
               "import {\n  View,\n  Text,\n  ScrollView,\n  StyleSheet,\n  Switch,\n  Alert,\n} from 'react-native';")
# Remove toggleModuleSwitch function
c = re.sub(r'\n\s*const toggleModuleSwitch = useCallback\([^)]*\);\s*\n', '\n', c)
# Remove moduleSwitches variable
c = re.sub(r"\s*const moduleSwitches = settings\?\.preferences\?\.moduleSwitches \?\? \{\};\s*\n", '\n', c)
with open(fp, 'w') as f: f.write(c)
print('FIXED: notify/index.tsx - removed TouchableOpacity, toggleModuleSwitch, moduleSwitches')

# --- Fix personal/about/index.tsx: remove unused 'Image' ---
fp = os.path.join(base, 'src/pages/personal/about/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("  TouchableOpacity,\n  Linking,\n  Image,\n} from 'react-native';",
               "  TouchableOpacity,\n  Linking,\n} from 'react-native';")
with open(fp, 'w') as f: f.write(c)
print('FIXED: personal/about/index.tsx - removed Image')

# --- Fix register/index.tsx: remove unused 'login' ---
fp = os.path.join(base, 'src/pages/register/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("  const login = useAuthStore(s => s.login);\n", '')
with open(fp, 'w') as f: f.write(c)
print('FIXED: register/index.tsx - removed login')

# --- Fix algorithm/index.tsx: remove unused 'ActivityIndicator' ---
fp = os.path.join(base, 'src/pages/system/algorithm/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("import { View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput, ActivityIndicator, Alert, RefreshControl } from 'react-native';",
               "import { View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput, Alert, RefreshControl } from 'react-native';")
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/algorithm/index.tsx - removed ActivityIndicator')

# --- Fix feedback/index.tsx: remove unused hasPerm, handleClose ---
fp = os.path.join(base, 'src/pages/system/feedback/index.tsx')
with open(fp) as f: c = f.read()
c = re.sub(r"\s*const hasPerm = useCallback\(\(p: string\) => perms\.includes\(p\), \[perms\]\);\s*\n", '\n', c)
c = re.sub(r"\s*const handleClose = \(item: FeedbackPageVO\) => \{[^}]*\}\);\s*\n", '\n', c)
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/feedback/index.tsx - removed hasPerm, handleClose')

# --- Fix member/detail.tsx: remove TouchableOpacity, TextInput ---
fp = os.path.join(base, 'src/pages/system/member/detail.tsx')
with open(fp) as f: c = f.read()
c = c.replace("import { View, Text, ScrollView, StyleSheet, TouchableOpacity, TextInput, Alert, ActivityIndicator } from 'react-native';",
               "import { View, Text, ScrollView, StyleSheet, Alert, ActivityIndicator } from 'react-native';")
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/member/detail.tsx - removed TouchableOpacity, TextInput')

# --- Fix member/growth-log.tsx: remove unused userId ---
fp = os.path.join(base, 'src/pages/system/member/growth-log.tsx')
with open(fp) as f: c = f.read()
if c.count('userId') <= 1:
    c = re.sub(r"\s*const \{ userId \} = route\.params;\s*\n", '\n', c)
    with open(fp, 'w') as f: f.write(c)
    print('FIXED: system/member/growth-log.tsx - removed unused userId')
else:
    print('SKIP: system/member/growth-log.tsx - userId might be used elsewhere')

# --- Fix member/index.tsx: remove unused hasPerm ---
fp = os.path.join(base, 'src/pages/system/member/index.tsx')
with open(fp) as f: c = f.read()
c = re.sub(r"\s*const hasPerm = useCallback\(\(p: string\) => perms\.includes\(p\), \[perms\]\);\s*\n", '\n', c)
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/member/index.tsx - removed hasPerm')

# --- Fix message/index.tsx: remove unused perms ---
fp = os.path.join(base, 'src/pages/system/message/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("  const perms = userInfo?.perms ?? [];\n\n", "\n")
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/message/index.tsx - removed perms')

# --- Fix order/index.tsx: remove unused hasPerm ---
fp = os.path.join(base, 'src/pages/system/order/index.tsx')
with open(fp) as f: c = f.read()
c = re.sub(r"\s*const hasPerm = useCallback\(\(p: string\) => perms\.includes\(p\), \[perms\]\);\s*\n", '\n', c)
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/order/index.tsx - removed hasPerm')

# --- Fix recommend/index.tsx: remove unused ActivityIndicator ---
fp = os.path.join(base, 'src/pages/system/recommend/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("import { View, Text, FlatList, StyleSheet, TouchableOpacity, ActivityIndicator, Alert, RefreshControl } from 'react-native';",
               "import { View, Text, FlatList, StyleSheet, TouchableOpacity, Alert, RefreshControl } from 'react-native';")
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/recommend/index.tsx - removed ActivityIndicator')

# --- Fix role/index.tsx: remove unused ActivityIndicator ---
fp = os.path.join(base, 'src/pages/system/role/index.tsx')
with open(fp) as f: c = f.read()
c = c.replace("  View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput,\n  ActivityIndicator, Alert, RefreshControl,\n} from 'react-native';",
               "  View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput,\n  Alert, RefreshControl,\n} from 'react-native';")
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/role/index.tsx - removed ActivityIndicator')

# --- Fix task/index.tsx: remove unused hasPerm ---
fp = os.path.join(base, 'src/pages/system/task/index.tsx')
with open(fp) as f: c = f.read()
c = re.sub(r"\s*const hasPerm = useCallback\(\(p: string\) => perms\.includes\(p\), \[perms\]\);\s*\n", '\n', c)
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/task/index.tsx - removed hasPerm')

# --- Fix user/form.tsx: remove unused Ionicons ---
fp = os.path.join(base, 'src/pages/system/user/form.tsx')
with open(fp) as f: c = f.read()
c = c.replace("import Ionicons from 'react-native-vector-icons/Ionicons';\n\n", "\n")
with open(fp, 'w') as f: f.write(c)
print('FIXED: system/user/form.tsx - removed Ionicons')

print('\nAll unused-vars fixes done!')
