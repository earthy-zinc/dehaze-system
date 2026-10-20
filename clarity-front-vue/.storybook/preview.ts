import type { Preview } from "@storybook/vue3-vite";
import { initialize, mswLoader } from "msw-storybook-addon";

/*
 * Initializes MSW
 * See https://github.com/mswjs/msw-storybook-addon#configuring-msw
 * to learn how to customize it
 */
initialize({
  onUnhandledRequest: "bypass",
});

const preview: Preview = {
  loaders: [mswLoader],
  parameters: {
    locale: "zh-CN",
    locales: {
      "zh-CN": { title: "中文", left: "🇨🇳" },
      en: { title: "English", left: "🇺🇸" },
    },

    toolbar: {
      locale: {
        icon: "globe",
        items: [
          { value: "en", title: "English" },
          { value: "zh-CN", title: "中文" },
        ],
      },
    },

    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },

    a11y: {
      // 'todo' - show a11y violations in the test UI only
      // 'error' - fail CI on a11y violations
      // 'off' - skip a11y checks entirely
      test: "todo",
    },
  },
};

export default preview;
