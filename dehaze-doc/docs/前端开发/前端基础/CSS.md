# CSS

## CSS简介

### CSS语法

```css
selector {
    property1: value1;
    property2: value2;
}
```

CSS语法由选择器和声明块组成：
- 选择器(selector)：指定要应用样式的元素
- 声明块：包含一个或多个声明，每个声明由属性和值组成

### CSS插入方式

CSS可以通过三种方式插入到HTML文档中：

| 插入方式 | 使用场景 | 实现方法 | 优先级 |
|---------|---------|---------|:-----:|
| 外部样式表 | 样式需要应用于多个页面 | 使用`<link>`标签引入样式表 | 低 |
| 内部样式表 | 样式仅应用于单个页面 | 使用`<style>`标签定义在HTML文件头部 | 中 |
| 内联样式 | 样式仅应用于单个元素 | 在元素中使用style属性 | 高 |

## CSS选择器

### 基本选择器

#### 元素选择器
根据标签名选择指定的一组元素
语法：`tag { }`

#### ID选择器
根据元素的id属性值选择指定的元素
语法：`#id { }`

#### 类选择器
根据元素的class属性值选择指定的一组元素
语法：`.class { }`

#### 通配选择器
选中当前页面所有元素
语法：`* { }`

### 复合选择器

#### 交集选择器
选择同时满足多个条件的元素
语法：`selector1selector2...selectorN { }`
注意：如果包含元素选择器，必须让元素选择器在最前面

#### 并集选择器
同时选择多个选择器对应的元素
语法：`selector1,selector2,...,selectorN { }`

### 关系选择器

#### 子元素选择器
选中指定父元素的直接子元素
语法：`element>element { }`

#### 后代元素选择器
选择指定元素内的所有指定后代元素（包括子元素及其子元素等）
语法：`element element { }`

#### 兄弟元素选择器
选择指定元素后面符合条件的兄弟元素
语法：
- `element+element { }`：选择紧邻的第一个符合条件的兄弟元素
- `element~element { }`：选择所有符合条件的兄弟元素

### 伪类选择器

针对超链接的不同状态可以使用以下伪类选择器：

| 选择器 | 说明 |
|-------|------|
| a:link | 未被访问过的链接 |
| a:visited | 已访问过的链接 |
| a:hover | 鼠标悬停在链接上时 |
| a:active | 鼠标点击时 |

### 常用选择器汇总

| 选择器 | 示例 | 说明 |
|-------|------|-----|
| .class | .intro | 选择所有class="intro"的元素 |
| #id | #name | 选择所有id="name"的元素 |
| * |  | 选择所有元素 |
| element | p | 选择所有`<p>`标签元素 |
| element,element | div,p | 选择所有`<div>`元素和`<p>`元素 |
| element element | div p | 选择所有`<div>`元素内的所有`<p>`元素 |
| element>element | div>p | 选择所有父级元素是`<div>`的`<p>`元素 |
| [attribute] | [target] | 选择所有带有target属性的元素 |
| [attribute=value] | [target=_blank] | 选择所有带有target属性且其值等于_blank的元素 |

## CSS属性

### 背景属性

| 属性 | 说明 | 示例 |
|-----|-----|-----|
| background-color | 设置背景颜色 | background-color: red; |
| background-image | 设置背景图像 | background-image: url('image.jpg'); |
| background-repeat | 设置背景图像的重复方式 | background-repeat: repeat/no-repeat/repeat-x/repeat-y; |
| background-attachment | 设置背景图像是否随页面滚动 | background-attachment: scroll/fixed; |
| background-position | 设置背景图像的位置 | background-position: center top; |

### 文本属性

| 属性 | 说明 | 示例 |
|-----|-----|-----|
| text-align | 设置文本水平对齐方式 | text-align: left/center/right; |
| text-decoration | 设置文本装饰线 | text-decoration: none/underline/overline/line-through; |
| text-indent | 设置文本首行缩进 | text-indent: 2em; |
| text-shadow | 设置文本阴影效果 | text-shadow: 2px 2px 5px gray; |
| color | 设置文本颜色 | color: blue; |
| line-height | 设置行高 | line-height: 1.5; |
| letter-spacing | 设置字符间距 | letter-spacing: 2px; |

### 字体属性

| 属性 | 说明 | 示例 |
|-----|-----|-----|
| font-family | 设置字体类型 | font-family: Arial, sans-serif; |
| font-style | 设置字体样式 | font-style: normal/italic; |
| font-size | 设置字体大小 | font-size: 16px; |

### 列表属性

可以对列表项的标记样式进行更改，默认情况下列表项会有项目符号。

### 表格属性

用于控制表格的外观样式，如边框、间距等。

### 边框属性

可以分别为元素的四个边设置边框样式：

| 属性 | 说明 | 示例 |
|-----|-----|-----|
| border-style | 设置边框样式 | border-style: solid/dashed/dotted; |
| border-width | 设置边框宽度 | border-width: 1px; |
| border-color | 设置边框颜色 | border-color: black; |
| border-radius | 设置圆角边框 | border-radius: 5px; |

### 轮廓属性

轮廓位于边框边缘的外围，主要用于突出显示元素：

| 属性 | 说明 | 示例 |
|-----|-----|-----|
| outline-style | 设置轮廓样式 | outline-style: solid/dashed; |
| outline-width | 设置轮廓宽度 | outline-width: 2px; |
| outline-color | 设置轮廓颜色 | outline-color: red; |

### 外边距属性

用于控制元素与其他元素之间的距离。

### 内边距属性

用于控制元素内容与其边框之间的距离。

### 尺寸属性

用于控制元素的大小：

| 属性 | 说明 | 示例 |
|-----|-----|-----|
| height | 设置元素高度 | height: 100px; |
| width | 设置元素宽度 | width: 200px; |
| line-height | 设置行高 | line-height: 24px; |
| max-height/min-height | 设置元素最大/最小高度 | max-height: 500px; |
| max-width/min-width | 设置元素最大/最小宽度 | min-width: 300px; |

### 可见性属性

| 属性 | 说明 |
|-----|-----|
| display: none | 隐藏元素，但仍占用原来的空间 |
| visibility: hidden | 隐藏元素，且不占用空间 |

### 定位属性

用于控制元素在页面中的位置。

### 盒子模型属性

控制内容溢出元素框时的表现：

| 属性 | 说明 | 示例 |
|-----|-----|-----|
| overflow-x | 控制内容在水平方向溢出时的处理方式 | overflow-x: visible/hidden/scroll/auto; |
| overflow-y | 控制内容在垂直方向溢出时的处理方式 | overflow-y: visible/hidden/scroll/auto; |

### 弹性盒子模型属性

| 属性 | 说明 | 示例 |
|-----|-----|-----|
| flex | 复合属性，设置弹性盒子模型的子元素如何分配空间 | flex: 1; |
| flex-grow | 设置扩展比率 | flex-grow: 1; |
| flex-shrink | 设置收缩比率 | flex-shrink: 1; |
| flex-basis | 设置伸缩基准值 | flex-basis: auto; |
| flex-direction | 定义主轴方向 | flex-direction: row/column; |
| flex-wrap | 控制换行行为 | flex-wrap: nowrap/wrap; |
| justify-content | 设置主轴对齐方式 | justify-content: center; |
| align-items | 设置交叉轴对齐方式 | align-items: center; |

## 盒子模型

CSS盒子模型从外到内包含以下几个部分：

![CSS box-model](https://www.runoob.com/images/box-model.gif)

1. **外边距(Margin)**：边框外的区域，外边距是透明的
2. **边框(Border)**：围绕在内边距和内容外的框架
3. **内边距(Padding)**：内容周围的区域，内边距是透明的
4. **内容(Content)**：盒子的内容，显示文本和图像