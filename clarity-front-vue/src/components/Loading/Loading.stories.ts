import Loading from "@/components/Loading/index.vue";
import type { Meta, StoryObj } from "@storybook/vue3-vite";

// More on how to set up stories at: https://storybook.js.org/docs/writing-stories
const meta = {
  title: "加载组件",
  component: Loading,
  tags: ["autodocs"],
  argTypes: {},
  args: {},
} satisfies Meta<typeof Loading>;

export default meta;
type Story = StoryObj<typeof meta>;
export const DefaultLoading: Story = {
  name: "加载中",
};
