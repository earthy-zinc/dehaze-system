import type { Meta, StoryObj } from '@storybook/react-native';
import Header from '../../src/components/Header';

const meta = {
  component: Header,
} satisfies Meta<typeof Header>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Basic: Story = {};
