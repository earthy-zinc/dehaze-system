import { defineClientConfig } from '@vuepress/client'

export default defineClientConfig({
    enhance({ router }) {
        let project = {
            "/kob": "82", // 90
            // alist 105 code-server 106
            "/dehaze-front": "83", //java 91 python 92
            "/pei-blog": "84", //93
            "/earthy-delivery/backend": "85/backend/index.html",
            "/earthy-delivery/front": "85/front/index.html",
            "/tuwei-mall/backend": "86", // 94
            "/tuwei-mall/front": "88",
            "/wpx-blog": "120",
            "/wpx-blog-admin": "121", // 122
        }

        router.beforeEach((to, from, next) => {
            if(typeof project[to.fullPath] === "string") {
                let redirect = new URL(window.location.protocol + "//"
                    + window.location.hostname + ":" + project[to.fullPath])
                window.open(redirect.toString());
                next(false);
            }
            next();
        })

        router.afterEach((to) => {
        })
    },
})
