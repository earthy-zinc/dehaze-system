# Spring Security 练习

## 1、概述

### 认证与授权

一般的Web应用都需要对用户进行认证和授权两个操作。

认证：Authentication 验证当前访问系统的用户是不是本系统的用户，并且需要确认具体是哪一个用户。

授权： Authorization 对经过认证后的用户判断它是否有权限进行某个操作

### 一般流程

1. 当用户登录时，前端将用户输入的用户名密码传输到后台，后台使用一个类对象将其封装起来，在Spring Security中使用的是`UsernamePasswordAuthenticationToken`类
2. 程序需要负责验证前端传来的这个对象，验证方法就是调用service服务——根据用户名从数据库中取出用户信息到实体类的实例中，然后比较两者密码，如果密码正确就成功登录。登录成功的同时把包含着用户信息如用户名、密码、用户具有的权限等信息的一个对象放到类似于Session的对象中（SecurityContextHolder）
3. 用户访问一个资源的时候，首先要判断该资源是否是限制访问的资源，如果是的化，需要判断当前用户是否登录，没有登录就跳转到登录页面。
4. 如果用户已经登陆，访问某一个限制访问的资源，程序要根据用户请求的地址（url）从数据库中取出该资源对应的所有可以访问该资源的角色，与当前用户所对应的角色一一对比，判断用户是否可以访问。

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-LKooVd3p-1648311184478)(C:\Users\30666\AppData\Roaming\Typora\typora-user-images\image-20220326160405082.png?lastModify=1648282919)]](https://img-blog.csdnimg.cn/993321379a3843f2bb0cb8df913acb2a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LiD5a-75YyX6YeM,size_20,color_FFFFFF,t_70,g_se,x_16)

用户提交的用户名和密码首先经过了一个认证过滤器，在这里将请求信息封装成一个类——Authentication类，这个类会存放用户信息，并且之后会存放用户的认证结果。然后调用认证方法到下一个地方进行认证。（也就是说判断是否是当前系统的用户）这里会通过一个认证管理器，认证管理器委托数据库访问认证提供者进行认证，这个提供者主要负责认证逻辑处理工作，首先从`UserDetailsService`用户详情服务商中调用`loadUserByUsername()`获取用户信息，会返回的用户详情信息类实例`UserDetails`，这个提供者通过密码加密器对比前端传来的密码和数据库中存在的密码是否一致。然后根据比对结果填充封装后的请求信息类。随后，过滤器便收到了认证的结果，过滤器将结果保存在服务器的一个叫做`SecurityContextHolder`的地方。通过这样一个方法`SecurityContextHolder.getContext().setAuthentication( authentication )`将认证结果保存。

部分方法说明：

1. `UsernamePasswordAuthenticationFilter`是处理用户名和密码认证方式的一个过滤器，将请求信息封装为Authentication类
2. `AuthenticationManager`是一个接口，定义了认证的方法，也是认证的入口，在这里他是用来管理各种认证方法的。有一个实现类ProviderManager里面会有一列表的认证方式。我们在自己写的登录认证Service中需要注入这样的接口，调用他的认证方法进行认证。会返回认证信息Authentication。我们可以对返回的认证信息进行后续的操作，如获取其中的用户信息构造jwtToken，然后将其存放在redis中
3. UserDetailsService接口：在这里我们需要指定相应的从数据库查询用户信息的方法。我们需要自己实现这个接口，注入一个DAO然后将其从数据库中查询到的用户信息转换成Security中要求的UserDetails对象即可。
4. UserDetails接口：这个接口存放了用户信息，以及与用户认证和授权有关的方法，Security会调用这个接口的实现类来判断账户的认证授权情况。

## 2、认证

## 3、授权

## 4、自定义失败处理

## 5、跨域

## 6、自定义权限校验方法

## 7、CSRF

## 8、认证处理器