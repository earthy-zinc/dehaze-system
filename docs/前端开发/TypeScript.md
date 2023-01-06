# 基础类型

## 基础类型表

| 类型名称       | 介绍                                                         | 示例                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| boolean        | 布尔值                                                       | let isDone: boolean = false;                                 |
| number         | 数字。Typescript中所有的数字都是浮点数，不仅支持十进制和十六进制，还支持二进制和八进制 | let dec: number = 6;                                         |
| string         | 字符串。使用双引号或者单引号表示字符串。模板字符串：被反引号包围，用`${expr}`这种形式嵌入表达式 | let name: string ="bob";<br>let sentence: string = `hello, ${name}` |
| list           | 数组。有两种方式可以定义，第一种再元素类型后面接上`[]`，第二种使用数组泛型`Array<元素类型>` | let list: number[] = [1, 2, 3];<br>let list: Array<Number> = [1, 2, 3]; |
| tuple          | 元组类型允许表示一个一致元素数量和类型的数组，各种元素的类型不必相同。 | let x: [string, number];<br>x = ["hello", 10];<br>x[0];      |
| enum           | 枚举类型，默认情况下从0为元素进行编号。                      | enum Color {Red, Green, Blue}<br>let c: Color = Color.Green; |
| any            | 任意类型，允许再编译时可选择性的包含或者移除类型检查         |                                                              |
| void           | 表示没有任何类型                                             |                                                              |
| Null/Undefined | 默认情况下这是所有类型的子类型                               |                                                              |
| never          | 表示那些永不存在值的类型，比如never类型是那些总是会抛出异常或者根本就不会有返回值的函数表达式或者箭头函数表达式的返回值类型。never类型是任何类型的子类型，也可以赋值给任何类型。但是没有类型可以赋值给never类型 |                                                              |
| Object         | 表示非原始类型                                               |                                                              |

## 类型断言

设定一个实体具有比他现有类型更加确切的类型。

```typescript
// 给定一个任意类型的变量。我们在使用这个变量是对类型进行断言，就可以使用该类型的相关方法
let someValue: any = "this is a string";
// 类型断言的尖括号写法
let strLength: number = (<string>someValue).length;
// 类型断言的as写法
let strLength: number = (someValue as string).length;
```

# 变量声明

# 接口

```typescript
interface LabelledValue {
    label: string;
    color?: string;
}
function printLabel(labelledObj: LabelledValue){
    console.log(labelledObj.label);
}
```

这是一个接口和函数的定义，在typescript中，为函数参数的冒号后面添加该参数对应的接口。就会调用typescript中的类型检查器对参数进行检查。

在这个例子中，我们在调用printLabel这个函数是需要传入一个对象参数，形参名为labelledObj，要求这个对象参数中一个名为label类型需要是string。我们传入的对象参数可能会包含很多属性，但是编译器只会检查那些必须的属性是否存在，并且类型是否匹配。而接口中的属性也不全都是必须的，有些只是在某些条件下存在，或者根本不存在，可选属性在应用

这里面接口就起到了给参数定义类型规范的作用。

# 模块

模块内的代码在其自身作用域中执行，而不是在全局作用域中。在模块内定义的变量、函数、类在模块外部是不可见的。不过我们可以通过export方式将需要其他模块使用的代码导出。其他模块就可以通过import进行导入。要想进行模块的导入和导出，我们需要用到模块加载器。这个加载器的作用就是在执行本模块代码之前查找并且导入这个模块所需要的所有依赖。

TypeScript中，包含顶级import或者export声明的文件都被当做一个模块。如果没有这个声明。则文件内容是全局可见的。

## 代码导出

任何代码声明，比如变量声明、函数声明、类、类型别名、接口。都可以通过export关键字进行导出。

```typescript
// 导出接口
export interface StringValidator {
    inAcceptable(s: string): boolean;
}
// 导出常量
export const numberRegexp = /^[0-9]+$/;
// 导出类
export class ZipCodeValidator implements StringValidator {
    isAcceptable(s: string) {
        return s.length === 5 && numberRegexp.test(s)
    }
}
// 导出前重命名
export {ZipCodeValidator as RegExpBasedZipCodeValidator}
// 集中导出——我们可以在一个模块内导入多个模块，然后将这些模块集中在该模块导出
export * from "./StringValidator";
```

## 代码导入

```typescript
// 导入一个模块中某个内容
import {ZipCodeValidator} from "./ZipCodeValidator"
// 导入并重命名
import {ZipCodeValidator as Validator} from c
// 将整个模块内容导出到一个变量中，然后通过变量访问这些内容
import * as validator from "./ZipCodeValidator"
validator.ZipCodeValidator();
```

### 默认导出

默认导出使用default标记。一个模块中只能有一个默认导出。标记为默认导出的类和函数的名字是可以省略的。导入该声明的模块需要自己定义默认导出的名称。

```typescript
// 使用export default关键字进行默认导出
export default class ZipCodeValidator {}
// 在导入时可以自定义默认导出的名称
import validator from "./ZipCodeValidator"
```

