---
order: -1
---

# LeetCode
```java
package org.pei;

import java.util.*;

public class Main {
    // 题目要求移除数组中所有的值为val的元素
    // 使用双指针，左指针执行下一个将要赋值的位置，右指针指向当前要处理的元素
    // 开始时左右指针都指向起始位置
    // 如果右指针指向的元素不是值val，则则说明我们需要保留该值，将该值复制到左指针位置，因为左指针是当前将要赋值的位置
    // 赋值完成后左右指针同时右移，左指针将指向下一个将要赋值的位置，右指针指向下一次要处理的元素
    // 如果右指针指向的元素是值val，则说明我们不要该值，右指针处理完毕，右移一位，而左指针无需赋值，因为当前值不是我们需要保留的
    // 直到遍历完成之后，左指针指向的仍是逻辑上下一个将要赋值的位置，其下标在逻辑上就对应移除val后数组的新长度
    public int removeElement(int[] nums, int val) {
        int n = nums.length;
        int left = 0;
        for (int right = 0; right < n; right++) {
            if (nums[right] != val) {
                nums[left] = nums[right];
                left++;
            }
        }
        return left;
    }

    // 方法1
    // 通过栈的先入后出的特点来解决左右括号匹配的问题
    // 遇到左括号则入栈，遇到对应的右括号时，则左括号出栈，如果遍历完所有括号后栈仍然为空
    //
    // 建立一个哈希表，用于构建左右括号之间的对应关系，key为左括号，value为右括号，查询两个括号只需要o(1)
    // 的时间复杂度，选择一个括号如果是左括号，则入栈，否则通过哈希表判断括号的对应关系，如果括号不对应的话，就提前返回false
    // 栈为空的时候，pop出栈会报错，那么给栈赋予初值 '?' ，当栈为空，并且当前字符为右括号时，就可以提前返回false
    private static final Map<Character, Character> map = new HashMap<>();

    static {
        map.put('{', '}');
        map.put('[', ']');
        map.put('(', ')');
        map.put('?', '?');
    }

    public boolean isValid(String s) {
        // 如果字符串的长度不为0，并且字符串首位字符就不包含三种左括号和问号，则直接返回
        if (!s.isEmpty() && !map.containsKey(s.charAt(0))) {
            return false;
        }
        LinkedList<Character> stack = new LinkedList<>();
        stack.add('?');
        // 从字符串中遍历出来的字符和哈希表中的键比较，也就是判断该字符是不是左括号，是的话就入栈，由于字符串中不存在"?"字符
        // 因此不会有？入栈
        // 否则的话，那就断定是一个右括号，获得栈中当前的左括号，判断哈希表中对应的右括号是否是当前的字符，如果不是则直接返回false
        // 再遍历结束后，会走到？匹配环节，此时会走到代码第二个return中，如果当前栈的大小正好为1，也就是只剩下了？，则返回true
        // 如果还存在其他字符，则栈的大小肯定不会是1
        for (Character c : s.toCharArray()) {
            if (map.containsKey(c)) {
                stack.addLast(c);
            } else if (!Objects.equals(map.get(stack.removeLast()), c)) {
                return false;
            }
        }
        return stack.size() == 1;
    }

    /**
     * 动态规划 对于一个子串而言，如果它是回文串，并且长度大于2，那么将其首尾的两个字母都去除之后，它仍然是一个回文串，
     * 比如字符串ababa，我们确定他是一个回文串，那么去除首尾的a，剩下的bab还是一个回文串，再去除首位的b，a本身也可以当作回文串 那么这种当前问题可以抛给子问题去解决，最终子问题有一个终点的情况，我们可以用动态规划来解决
     * 用P(i, j)来表示字符串s的第i到j个字母组成的串。那么这个串是一个回文串的条件就是， 如果串i-1, j+1是一个回文串，并且i-1和j+1位置上的字符是相同的，那么才会是一个回文串
     * <p/>
     * 边界条件是字串长度小于等于2 如果字串长度为1，显然是一个回文串，对于长度为2的字串，只要它的两个字母相同那么就是一个回文串。 那么动态规划的边界条件就是
     */
    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2) return s;
        // 定义最大字串长度至少从1开始
        int maxLen = 1;
        int begin = 0;
        // dp[i][j] 表示 从i到j组成的字符串是否是回文串
        boolean[][] dp = new boolean[len][len];
        for (int i = 0; i < len; i++) {
            // 所有的长度为1的都是字串
            dp[i][i] = true;
        }
        char[] charArray = s.toCharArray();
        // 然后从字串长度为2开始循环到字串长度等于其自身
        for (int L = 2; L <= len; L++) {
            // 枚举字串的左边界，字串左边界从下标为0开始
            for (int i = 0; i < len; i++) {
                // 确定有边界，右边界的确定应该以字串长度为L = j - i + 1
                int j = L + i - 1;
                // 如果右边界越界了，就退出当前循环
                if (j >= len) break;
                // 否则判断左右两个边界的字符是否是一样的，如果不是一样的，那么当前字符串肯定不是一个回文串
                if (charArray[i] != charArray[j]) {
                    dp[i][j] = false;
                } else {
                    // 如果是一样的话，那么要看舍去了左右两个字符后的字串是否是回文串，如果是，则该字串也是回文串
                    // 如果不是，则该字串也不是回文串，实际上用一条语句就可以说明，也就是该字串是否是回文串，取决于
                    // 其字串是否是回文串，这种情况下其字串至少存在，如果只有两个字符，是没有舍去左右字符的字串的
                    // 边界情况是该字串只有两个字符，则该字串是回文串
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                // 然后计算回文串的最大长度。最大回文串的起始字符下标
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }

    /**
     * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。 请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
     * <p/>
     * 首先按照数组的左端点进行排序，那么在排完序的列表中，可以和并的区间一定是连续的，那么从前面两个开始比较，
     * 如果如果发现第二个数组的下界小于等于第一个数组的上界，那么这两个数组就可以合并，以最小的下界和最大的上界作为合并后数组的上下界
     * <p>
     * 我们用数组merge存储最终的答案，首先将列表中的区间按照左端点升序排序，使用Arrays.sort方法可以快速进行数组之间的排序。 将第一个区间加入merged数组中，然后按照顺序考虑之后的区间
     */
    public int[][] merge(int[][] intervals) {
        // 如果第一个数组的左端点大于第二个数组的左端点，则返回true，将大的排到小的后面
        Arrays.sort(intervals, Comparator.comparingInt(interval -> interval[0]));
        List<int[]> merged = new ArrayList<>();
        for (int[] interval : intervals) {
            // 获取当前区间的上界和下界
            int left = interval[0];
            int right = interval[1];
            // 在实际中我们这样考虑，如果merge中存放上一个区间的的下界小于当前区间的上界，说明这两个区间不能合并
            // 当然要考虑merge中没有数组的情况
            if (merged.isEmpty() || merged.get(merged.size() - 1)[1] < left) {
                merged.add(interval);
            } else {
                // 另一种情况我们要考虑合并这两个数组
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], right);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }

    // 设置两个指针，左指针和右指针
    // 左指针记录当前出现的首个最大的元素的位置
    // 右指针则往前移动，寻找之后比左指针上元素大的第一个元素
    // 如果右指针当前元素等于左指针上的元素值，则右指针+1
    // 如果右指针当前元素大于左指针上的元素，且右指针的下标大于左指针下标+1，则将右指针上的元素赋值到左指针下标+1的位置上，右指针+1，左指针+1
    // 如果右指针指向超过数组下标，说明在左指针的右边，不再存在比左指针上的元素值更大的元素了，则左指针则为数组的新下表，左指针+1则为数组中唯一元素的个数
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        int currentMax;
        int left = 0;
        for (int right = 0; right < n; right++) {
            currentMax = nums[left];
            if (nums[right] > currentMax) {
                if (right > left + 1) nums[left + 1] = nums[right];
                left++;
            }
        }
        return left + 1;
    }

    /**
     * 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。 <p/>判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
     * 我们从第一个下标开始跳跃，判断能否最终到达最后一个下标上，我们可以拓展一下，先判断对于数组中的任意一个位置，要如何判断 它是否可以到达，只要存在一个位置x，它本身是可以到达的并且跳跃到它的最大长度是x + nums[x]
     * 这个值大于等于y，那么位置y也是可以到达的 换句话说，对于每一个可以到达的位置x，它会使得x+1，x+2.到x+nums[x]这些连续的位置都可以到达
     * 也就是说我们到达了位置x，而位置x上的值为nums[x]这个值代表了则在位置x我们可以跳跃的最大长度 所以位置x+1，x+2.到x+nums[x]这些位置我们都可以通过在位置x上跳跃到达的。
     * 这样以来我们依次遍历数组中的每一个位置，并且实时维护最远可以到达的位置，对于当前遍历到的位置，如果他在最远可以到达的范围内 那么我们就可以从起点通过一步步的跳跃到达该位置，因此我们用x+nums[x]来更新当前最远能到达位置
     * 在遍历的过程中，如果最远可以到达的位置大于等于数组中的最后一个位置，则说明最后一个位置是可以到达的，返回true， 反之在遍历结束后最后一个位置仍然不可以到达就返回false
     */
    public boolean canJump(int[] nums) {
        // 表示当前所能到达的位置，0即只能到达初始位置
        int maxReached = 0;
        for (int i = 0; i < nums.length; i++) {
            // 如果当前所在的位置小于等于之前所能到达的最大位置，则要判断一下
            // 之前所能到达的最大位置和当前位置上所能到达的最大位置哪一个大
            // 取出最大的作为下一阶段所能到达的最大位置
            if (i <= maxReached) {
                maxReached = Math.max(maxReached, i + nums[i]);
                if (maxReached >= nums.length - 1) {
                    return true;
                }
            }
            // 如果当前位置大于所能到达的最大位置，则什么也不做，不能提前返回false
        }
        return false;
    }


    /**
     * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。 请你将两个数相加，并以相同形式返回一个表示和的链表。 你可以假设除了数字 0 之外，这两个数都不会以 0
     * 开头。
     */
    public static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    /**
     * 个位数先加，逢十进一 当出现一个链表走到头时，则将另一个链表之后的数字直接复制到新链表中（当然有进位需要加一下）
     */

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        int val = 0;
        ListNode l3 = new ListNode();
        ListNode preNode = l3;
        ListNode curNode;
        while (l1 != null && l2 != null) {
            int sum = l1.val + l2.val + carry;
            // 当前的进位数，不会超过10，需要加上上一步的进位数
            carry = sum / 10;
            // 当前位的数值，当前位的两数加和减去进位的数
            val = sum % 10;
            // 创建当前的结点并赋值
            curNode = new ListNode(val);
            // 链接到前一个结点上
            preNode.next = curNode;
            // 当前结点变为前一个结点
            preNode = curNode;
            // 两数向前移
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l1 != null) {
            int sum = l1.val + carry;
            carry = sum / 10;
            val = sum % 10;
            curNode = new ListNode(val);
            preNode.next = curNode;
            preNode = curNode;
            l1 = l1.next;
        }
        while (l2 != null) {
            int sum = l2.val + carry;
            carry = sum / 10;
            val = sum % 10;
            curNode = new ListNode(val);
            preNode.next = curNode;
            preNode = curNode;
            l2 = l2.next;
        }
        if (carry != 0) {
            preNode.next = new ListNode(carry);
        }
        return l3.next;
    }

    public ListNode addTwoNumbersRevisited(ListNode l1, ListNode l2) {
        ListNode l3 = new ListNode();
        ListNode preNode = l3;
        int carry = 0;
        // 当l1或l2不空时进行循环
        while (l1 != null || l2 != null) {
            // 我们要从两个链表中分别取出一个值，进行相加操作，然后取其余数当作当前位的值
            // 取除数的证书部分作为进位的值
            int n1 = l1 != null ? l1.val : 0;
            int n2 = l2 != null ? l2.val : 0;
            int sum = n1 + n2 + carry;
            ListNode curNode = new ListNode(sum % 10);
            preNode.next = curNode;
            preNode = curNode;
            carry = sum / 10;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
        if (carry != 0) {
            preNode.next = new ListNode(carry);
        }
        return l3.next;
    }

    /**
     * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度 通过滑动窗口法求解，首先找一下第一个字符开始的最长字符串，找出当前最长的
     * 字符串的长度，然后在从下一个字符开始，找出以该字符位起始点的最长字符串，一致到最后一个字符串 依次递增地枚举字串的起始位置，字串的结束位置也是递增的，选择字符串的第k个字符作为位置
     * 那么我们会得到不包含重复字符的以第k个字符为起始位置的最长字串的结束位置为rk，这中间是连续的 那么当我们选择第k+1个字符作为起始位置时，首先从k+1到rk的字符是不重复的，并且由于少了原本的
     * 第k个字符，我们可以继续增大rk直到右侧出现重复字符为止 我们用两个指针表示字符串中的某个字串的左右边界，其中左指针代表这上文中枚举字串的起始位置，而右指针为rk
     * 从左指针开始，找最长不连续字串，向右移动右指针，保证两个指针之间不存在重复字符，在遇到重复字符后，则断掉当前 的循环，与之前的最大字串长度比较，然后记录当前最大字串长度，随后移动左指针重新开始循环
     * 在枚举结束之后，我们就能够找到最大字串长度 那么如何判断重复字符 我们使用哈希表来判断重复字符，哈希表中存放的元素是不重复的，我们在循环中将当前字符存放到哈希表中，
     * 进一步在左指针向右移动时，我们从哈希表中移除左指针所指向的元素，在右指针向右移动时，我们往哈希表中加入右指针指向的元素
     */
    public int lengthOfLongestSubstring(String s) {
        // 右指针移动后应该把当前位置的元素加入哈希表，因此右指针一开始不能指向第一个元素，需要移动赋值
        int right = -1;
        Set<Character> hash = new HashSet<>();
        // 表示当前最长连续字串
        int answer = 0;
        int n = s.length();
        for (int left = 0; left < n; left++) {
            // 当右指针的下一个元素不越界，并且哈希表中不包含下一个元素的值，则将该元素加入哈希表
            while (right + 1 < n && !hash.contains(s.charAt(right + 1))) {
                hash.add(s.charAt(right + 1));
                right++;
            }
            answer = Math.max(answer, right - left + 1);
            // 移除左指针当前指向的元素，为下一次循环做准备
            hash.remove(s.charAt(left));
        }
        return answer;
    }

    /**
     * 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，（这里指的是下标不重复）
     * 但是对下标的顺序没有要求，也就是我们可以通过排序打乱下标，只要两个数不是在同一个下标下面就行。同时还满足 nums[i] + nums[j] + nums[k] == 0 。对应下标位置上的值的和为0 你返回所有和为 0
     * 且不重复的三元组。
     * <p>
     * 从左到右，先从第一个数开始，选中这个数作为三元组的第一个数，然后从数组后面找寻两个数，这两个数的和应该是第一个数的值的负数。
     * 但是，用简单的三重循环，得到的时间复杂度为N3，而我们还需要使用哈希表进行去重操作，得到不包含重复三元组的最终答案，
     * 如果我们固定了前两重循环得到的元素，那么在第三次循环时只有唯一的元素能满足元素之和为0，我们将整个数组排序，在第二重循环
     * 向后枚举下一个元素时，那么下一个元素必然大于上一个元素，那么之后第三个元素就小于之前第三个元素，那么现在的第三个元素
     * 是出现在之前第三个元素的左侧，我们可以从小到大枚举b，同时从大到校枚举c，那么第二重循环和第三重循环其实是并列的关系
     * 那么这就是双指针，如果我们需要枚举数组中两个元素，我们发现随着第一个元素递增，而第二个元素是在递减的，那么可以使用双指针方法。
     * 将两重循环的时间复杂度从n2降低到n，在枚举过程的每一步中，左指针会向右移动一个位置，而右指针会向左移动若干个位置，但是他们一共会移动的位置数最大不超过N的数量级。
     */
    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for (int first = 0; first < n; first++) {
            // 在进行下一次枚举时，第一个数需要和上次循环的第一个数值不同，如果相同的话，说明枚举重复
            // 同时注意边界调价，在第一次循环开始时我们不能跳过
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // 第三个数，也就是右指针一开始需要指向数组的最右端
            int third = n - 1;
            // 第二个数和第三个数的和应是第一个数的负数
            int target = -nums[first];
            for (int second = first + 1; second < n; second++) {
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 在第二重循环中，第二个数固定，找到能使第二第三个数的和能等于第一个数的负数那种情况
                // 而排序从左到右数依次增大，而第三个数使最右侧的，所以很有可能出现二三数之和大于第一个数负数情况
                // 同时第三个数不能小于第二个数，因为要保证数不能有重复，因此将右指针逐步往右移动
                while (second < third && nums[second] + nums[third] > target) {
                    third--;
                }
                //  如果指针second 和third重合。则说明本次循环没有能使第二第三个数的和能等于第一个数的负数那种情况
                // 同时说明，当前循环的second和third是总循环中的最小值了，而最小值仍然有nums[second] + nums[third] > target
                // 我们后续是找不到满足条件的值了
                // 直接跳过该循环
                if (second == third) {
                    break;
                }
                // 如果第二第三个数的和能等于第一个数的负数，那么就将这几个数加入三元组中
                // 就此停止寻找第三个数，因为如果有右指针在往左走，第三个数必然需要和上一个第三个数值相等才能满足要求
                // 而这样就会造成三元组重复
                if (nums[second] + nums[third] == target) {
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    /**
     * 给你一个 无重叠的 ，按照区间起始端点排序的区间列表。 在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。
     * <p/>
     * 在给定的区间集合互相不重叠的情况下，如果我们需要插入一个新的区间 1. 找出所有与区间S重合的区间集合 2. 将这个区间集合中所有区间连带上区间S合并成一个大区间 3.
     * 最终答案即不与原区间集合重叠的区间以及合并后的大区间组成的集合。 在给定区间集合已经按照左端点排序的前提下，所有与区间S重叠的区间在数组中下标范围是连续的我们对所有区间进行一次遍历，就可以找到这个连续的下标范围 1.
     * 定义新区间为S = [left, right]当我们遍历到区间li, ri时，如果ri < left, 当前区间的右端点小于新区见的左端点，说明当前区间与新区间
     * 是不重叠的，我们可以直接将**当前区间**加入答案，这是一个独立的区间 2. 如果li > right，当前区间的左端点大于新区间的右端点，说明当前区间和新区间是不重叠的，直接将**当前区间**加入答案 3.
     * 如果上述两种情况都不满足,说明新区间是和当前区间有重叠的，我们就需要将当前区间和新区间进行合并，将新区间的左右更新为与新区间的并集
     * <p/>
     * 最后就是判断什么时候将区间S加入答案 我们需要保证加入新区间后，列表内的区间仍然是有序的，意味着答案也是按照左端点排序的，因此当我们遇到第一个满足左端点大于新区间的右端点
     * 时，说明以后遍历道德区间不会与新区间产生重叠，并且他们的左端点一定大于新区间的右端点，此时我们可以将S加入到答案中。
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        // 定义新区间的左右端点值
        int left = newInterval[0];
        int right = newInterval[1];
        List<int[]> ansList = new ArrayList<>();
        boolean placed = false;
        for (int[] interval : intervals) {
            int li = interval[0];
            int ri = interval[1];
            if (li > right) {
                // 当前区间在新区间的右侧并且没有交集，并且之前的区间在上次循环时已经检验过不会有交集，则直接将新区间插入
                if (!placed) {
                    ansList.add(new int[]{left, right});
                    placed = true;
                }
                ansList.add(interval);
            } else if (ri < left) {
                ansList.add(interval);
            } else {
                left = Math.min(left, li);
                right = Math.max(right, ri);
            }
        }
        if (!placed) {
            ansList.add(new int[]{left, right});
        }
        return ansList.toArray(new int[ansList.size()][2]);
    }

    /**
     * 给定一个含有 n 个正整数的数组和一个正整数 target 。 找出该数组中满足其总和大于等于 target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，
     * 并返回其长度。如果不存在符合条件的子数组，返回 0 。
     * <p/>
     * 方法1：给定一个循环，从第一个数开始连续往后加，直到其总和大于等于target，则是从该数开始的总和大于等于 target 的长度最小的 连续子数组，记录长度 最大时间复杂度为n2
     * 方法2：由于数组中只有正整数，找连续数组，我们可以使用前缀和实现，前缀和也就是从下标0开始，每次往后增加一个元素，求和，最后得到一个和原数组
     * 长度相等的数组，其中数组的元素sum[i]就表示nums[0]到nums[i-1]的元素和，这样的话，前缀和是一个递增的有序序列，我们可以使用二分查找。
     * 得到前缀和之后，对于原数组的每一个开始下标，我们可以通过二分查找的到大于或者等于i的最小下标bound，使得sums[bound] - sums[i-1] >= s 并且更新子数组的最小长度，子数组的长度为 bound -
     * (i-1)
     */
    public int minSubArrayLen(int target, int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        int answer = Integer.MAX_VALUE;
        int[] sums = new int[n + 1];
        /*
         为了方便计算，sum[0]表示前0个元素的前缀和为0，sums[1]表示前1个元素的前缀和
         */
        for (int i = 1; i <= n; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
        for (int i = 1; i <= n; i++) {
            // 目标值应该是总体目标值加上上一步的前缀和
            // 在原数组中接近target的最短连续数组，转换到前缀和上的target应该是原target加上上一步的前缀和，这样才能和之前的对应
            // 或者说，查找大于等于开始下标i的最小下标，使得sums[bound] - sum[i-1] >= target
            // sums[bound] - sum[i-1]这一段表示就是原数组从i到bound的这一段连续区间的和，我们要找到哪个最小的下标bound
            // 使得这个连续区间的和刚好大于等于题目给的值target
            // 我们可以将不等式变换，有 sums[bound] >= target + sum[i-1] 设sums[bound] = curTar
            // 则我们要在sums数组中二分查找最接近curTar的下标
            int curTar = target + sums[i - 1];
            /*
            java的二分查找。在找到完全匹配的值时，会返回对应的数组下标。
            而在没有找到完全匹配的值时，会返回一个负数，我们取（-（插入点）-1）。
            插入点定义为将curTar插入数组的点：大于curTar的第一个元素的索引，或者如果数组中的所有元素都小于curTar，则定义为a.length。
             */
            int bound = Arrays.binarySearch(sums, curTar);
            if (bound < 0) {
                // 获取比curTar大的第一个数的索引
                bound = -bound - 1;
            }
            if (bound <= n) {
                // 比较上一步找到的最小数组长度和当前找到的最小数组长度，取两者最小值
                answer = Math.min(answer, bound - (i - 1));
            }
        }
        return answer == Integer.MAX_VALUE ? 0 : answer;
    }

    /**
     * 定义两个指针分别表示子数组的开始位置和结束位置，维护变量sum存储子数组中元素的和，也就是从num[start]到num[end]的元素和 初始状态下start和end都指向0，sum的值为0
     * 每一轮迭代，将nums[end]加到sum中，如果sum>=target则更新子数组的最小长度，此时子数组的最小长度为end - start + 1 然后将nums[start]
     * 从sum中减去然后右移，直到sum<s，在此过程中更新子数组最小长度在每一轮迭代最后，将end右移
     */
    public int minSubArrayLenRevisit(int target, int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        int answer = Integer.MAX_VALUE;
        int start = 0;
        int end = 0;
        int sum = 0;
        while (end < n) {
            sum += nums[end];
            while (sum >= target) {
                answer = end - start + 1;
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return answer == Integer.MAX_VALUE ? 0 : answer;
    }

    /**
     * 给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。 每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。
     * 换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j]处:
     * 0 <= j <= nums[i] , i + j < n 返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
     * <p/>
     * 方法1：反向查找出发位置
     * 目标是到达数组的最后一个位置，因此我们可以考虑最后一步跳跃前所在的位置，在这样的位置中可以通过跳跃到达最后一个位置
     * 如果有多个位置都能够通过跳跃到达最后一个位置，那么我们可以使用贪心算法选择距离最后一个位置最远的哪一个，也就是
     * 对应下标最小的哪个位置，我们可以从左到右遍历数组，选择第一个满足要求的位置，找到之后再继续使用贪心算法，找倒数第二步跳跃前的位置
     * 直到找到数组的开始位置
     */
    public int jump(int[] nums) {
        // 当前跳跃所在的位置
        int position = nums.length-1;
        // 跳跃需要用到的步数
        int step = 0;
        while(position > 0){
            for(int i = 0; i< position; i++){
                //如果循环所在的位置+可跳跃的部署大于当前跳跃应在的位置
                if(i + nums[i] >= position){
                    position = i;
                    step++;
                    break;
                }
            }
        }
        return step;
    }

    /**
     * 如果我们贪心的进行正向查找，每次找到可以到达的最远位置，就可以在线性时间内得到最少的跳跃次数。
     * 也就是说我们从下标0出发，可以跳到下标1、2等几个位置，而我们从能挑到的这几个位置中选择从这几个位置中能跳的更远的哪一个
     * 在具体的实现中，我们维护当前能够到达的最大下标位置，记为边界，从左到右遍历数组，到达边界时，更新边界并且将跳跃次数增加1
     * 在遍历数组时，我们不访问最后一个元素，因为在访问最后一个元素之前，我们的边界一定大于等于最后一个位置。
     */
    public int jumpRevisited(int[] nums){
        int length = nums.length;
        // 当前通过二次跳跃最远能够到达的边界位置
        int end = 0;
        // 当前元素跳跃能够到达的最大下标的位置
        int maxPosition = 0;
        int step = 0;
        for(int i = 0; i<length-1; i++){
            // 当前能到达的最大位置，
            maxPosition = Math.max(maxPosition, i + nums[i]);
            // 如果当前位置已经是上次记录的通过二次跳跃最远能够到达的边界位置
            // 那就将通过二次跳跃最远能够到达的边界位置更新为当前能到达的最大位置，然后步数+1相当于进行了一次跳跃
            if(i == end){
                // 之前所能到达的最大位置
                end = maxPosition;
                step++;
            }
        }
        return step;
    }

    /**
     * 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i])
     * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
     * 返回容器可以储存的最大水量。
     * 双指针法，在初始时，左右指针分别指向数组左右两端，可以容纳的水量就为左右指针指的最小值*宽度
     * 然后移动对应数值较小的哪个指针指针，我们另外搞一个值记录盛水最多的最大水量，每当水量比之前的大时，就更新这个值
     */
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int max = 0;
        while(left != right){
            max = Math.max(Math.min(height[right], height[left]) * (right - left), max);
            if(height[left] >= height[right]){
                right--;
            }else {
                left++;
            }
        }
        return max;
    }

    /**
     * 给你两个字符串 haystack 和 needle ，
     * 请你在 haystack 字符串中找出 needle 字符串的第一个匹配项的下标（下标从 0 开始）。
     * 如果 needle 不是 haystack 的一部分，则返回  -1 。
     */
    public int strStr(String haystack, String needle) {

        return 0;
    }

    /**
     * 给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。
     * 在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。
     * 返回 你能获得的 最大利润 。
     * 方法1:暴力搜索
     * 由于不限制交易次数，在每一天都可以根据当前是否持有股票选择响应的操作。
     * 在第一天时，我们可以选择买入或者不买入，第二天就分为卖出和不卖出，第三天就继续延伸出买入、卖出、是否继续买入。不买入、买入、是否卖出
     * 也就是出现了二叉树的情况。我们可以通过深度优先搜索获取最大利润所在的那一个分支
     */
    public int res;
    public int maxProfitViolent(int[] prices) {
        int len = prices.length;
        // 如果只有1天时间，买入后无法卖出，也就无法获取收益
        if(len < 2) return 0;
        this.res = 0;
        dfs(prices, 0, len, 0, res);
        return this.res;
    }

    /**
     *
     * @param prices 股价数组
     * @param index 当前是第几天
     * @param len 数组长度
     * @param status 当前是否持有股票，0表示不持有
     * @param profit 当前收益
     */
    private void dfs(int[] prices, int index, int len, int status, int profit){
        // 当前数组下标等于数组长度时，说明已经到了递归终点，则判断之前存储的最大收益和当前收益哪一个大
        if(index == len){
            this.res = Math.max(this.res, profit);
            return;
        }
        // 深度有限，先一直进入下一天，一直到最后一天，都不持有股票的情况
        dfs(prices,index+1,len,status,profit);
        if(status == 0) {
            // 上一天不持有股票，这次尝试持有股票，收益减少
            dfs(prices, index + 1, len, 1, profit - prices[index]);
        } else {
            //上一天持有股票，这次尝试不持有，收益增加
            dfs(prices, index + 1, len, 0, profit + prices[index]);
        }
    }

    /**
     * 动态规划，
     */
    public int maxProfit(int[] prices) {
        return 0;
    }

    /**
     * 给你两个二进制字符串 a 和 b ，以二进制字符串的形式返回它们的和。
     * 末尾对齐，逐位相加。再我们可以区n为a的绝对值和b的绝对值中较大的哪一个，循环n次，从最低位开始遍历
     * 使用变量carry表示上一个位置的进位，初始值为0，记录当前位置对其的两个为ai和bi，则每一位的答案就为
     * (carry + ai + bi) mod 2，下一位的进位式 (carry + ai + bi ) / 2向下取整，重复上述步骤
     * 直到数字a和数字b的每一位都计算完毕，最后如果carry的最高位不为0，则需要将最高位添加到计算结果的末尾
     * 为了让各个位置对齐，我们需要先反转这个代表二进制数字的字符串，然后低位下标对应地位，高位下标对应高位
     * 然后从高位向低位遍历，对应位置的答案按照顺序存入答案字符串。
     *
     */
    public String addBinary(String a, String b) {
        StringBuffer ans = new StringBuffer();
        int n = Math.max(a.length(), b.length());
        int carry = 0;
        for(int i = 0; i < n; i++){
            if(i < a.length()){
                // 相当与ascii码之间相减
                carry += a.charAt(a.length() - 1 - i) - '0';
            }
            if(i<b.length()){
                carry += b.charAt(b.length() - 1 - i) - '0';
            }
            // 也就是获得当前位的答案，当前的carry就已经加过了a和b 第i位上的数字了，然后再把之前减去的字符0加上
            // 字符'0'在这里的作用是将字符转化为整数，但是数字字符对应的ascii码整数值并不是从0开始的，所以减去’0‘之前的整数
            // 以防影响计算
            ans.append((char) (carry % 2 + '0'));
            carry /= 2;
        }
        // 在循环完成后，如果还有进位值，则再加1
        if(carry > 0){
            ans.append('1');
        }
        ans.reverse();
        return ans.toString();
    }

    /**
     * 颠倒给定的 32 位无符号整数的二进制位。
     * 位运算& 如果响应位都是1，则结果为1，否则为0
     * 位运算| 如果相应为都是0，则结果是0，否则为1
     * 位运算^ 如果相应位的值相同，则结果位0，否则为1
     * 位运算~ 按照位取反操作，即如果原位是0，则变为1，如果原位是1，则变为0
     * << 按位左移运算，左操作数按位左移指定的位数
     * >> 按位右移运算，左操作数按照位右移指定的位数
     * >>> 按位右移补零操作符左操作数的值按照有操作数的指定位数右移，移动后之前位置的空位以0填充
     * 在Java中int是32位的，我们将n视作一个长度位32的二进制串，从低位到高位枚举n的每一位，将其倒叙添加到翻转的结果rev中
     * 每次枚举一位后，就将n右移一位，这样当前n的最低位就是我们要枚举的比特位，当n== 0的时候便可以结束循环
     */
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int reversed = 0;

        for(int i = 0;i<32 && n!=0; i++){
            // 1的比特位是00000000000000000000000000000001，也就是只看最后一位，如果最后一位为1，则为1
            // 然后我们将结果左移 31 - i位也就是颠倒了这一位，接下来是如何将这一位放置到结果上
            // 将reversed 与计算结果进行位运算 | 这个运算符的结果，因为上一步计算结果其他位都为0，所以如果是0的话应该不能影响运算的结果
            reversed |= (n & 1) << (31 - i);
            n >>>= 1; // 将n按位右移一位然后重新赋值为n
        }
        return reversed;
    }

    /**
     * 编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。
     *
     */
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int count = 0;
        for(int i = 0; i<32 && n!=0; i++){
            count += (n & 1) == 1 ? 1 : 0;
            n >>>= 1;
        }
        return count;
    }

    /**
     * 给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
     * 你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。
     * 如果不考虑时间复杂度和空间复杂度
     * 1. 使用集合来存储数字，遍历数组中的每个数字，如果集合中没有该数字，则将该数字加入集合，如果集合中已经有该数字，则将该数字从集合中删除
     *     最后剩下的就是只存在一次的数字
 *   * 2. 使用哈希表来存储每个数字和该数字出现的次数，遍历数组既可以得到每个数字出现的次数，并且更新哈希表。最后遍历哈希表，得到只出现一次的数字
     * 3. 使用集合存储数组中出现的所有数字，并且计算数组中所有元素之和，由于集合保证元素无重复，因此计算集合中的所有元素之和的两倍
     *
     * 我们使用异或运算 ^ 对应为的值相同，则为0，否则为1 ， 1^1 => 0, 0^0 => 0, 1^0 => 1, 0^1 => 1
     * 任何数和0进行异或运算 结果仍然是原来的数 0^0 => 0, 1^0 => 1,
     * 任何数与其自身做异或运算，结果为0 1^1 => 0, 0^0 => 0,
     * 异或运算妈祖交换律和结合率
     * <p/>
     * 我们假设数组中有 2m + 1个数，其中有m个数各出现两次，一个数出现一次，令a1 a2 am 为出现两次的m个数，am+1 为出现一次的数
     * 那么a1 到 am 这些数，和自身异或运算的结果都是 0 ，再与只有一个数的am+1进行异或，则等于am+1
     * 因此，将数组中的全部元素进行异或运算的结果就是数组中只出现一次的数字
     */
    public int singleNumber(int[] nums) {
        int single = 0;
        for(int num : nums){
            single ^= num;
        }
        return single;
    }

    /**
     * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
     * 动态规划。
     * 我们定义dp[i]表示字符串s前i个字符组成的字符串s[0....i-1]是否能够被空格拆分成若干个字典中出现的单词
     * 从前往后计算考虑转移方程，每次转移的时候我们需要枚举包含位置i-1的最后一个单词，看他是否出现在字典中。
     * 以及去除这部分字符串是否合法。
     * 公式化来说，我们需要枚举s[0... i-1]中的分割点j，看s[0...j-1]组成的字符串s1，和s[j...i-1]
     * 组成的字符串是否都合法，如果两个字符串军合法，那么按照定义s1 s2拼接成的字符串也同样合法。
     * 由于计算到dp[i]时，我们已经计算除了dp[0, i-1]的值，因此字符串s1是否合法可以直接由dp[j]得知，
     * 剩下的，我们只需要看s2是否合法即可，因此我们可以得出如下的结果
     * dp[i] = dp[j] && check(s[j...i-1])
     * 其中check函数表示字串是否出现在字典中
     * 对于检查一个字符串是否出现在给定的字符串列表里一般可以考虑哈希表快速判断，同时也可以做一些简单的
     * 剪枝操作，枚举分割点的时候倒着枚举，如果分割点j到i的长度已经大于字典列表里最长的单词长度，那么就结束枚举，
     * 对于边界条件 dp[0] = true
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        // 首先对字符串列表字典进行去重，之后就可以快速判断对应的字符串是否存在在字符串列表中
        Set<String> wordDictSet = new HashSet<>(wordDict);
        // dp[i] 表示字符串 s 前 i 个字符组成的字符串 s[0..i−1] 是否能被空格拆分成若干个字典中出现的单词
        boolean[] dps = new boolean[s.length() + 1];
        dps[0] = true;
        // 遍历i确定dps[1....i]的值， 而本身dps[i]的值需要dp[j]来确定，j需要从0开始遍历字串确定
        for(int i = 1; i<=s.length(); i++){
            // 如果字符串 s 前 j 个字符组成的字符串能拆分为字典中的单词 也就是dps[j] == true
            // 并且之后的j-i这部分的字串也在字典中
            // 那么我们就断定 dps[i] == true，也就是前i个字符组成的字符串都能拆分为字典中的单词
            for(int j = 0; j<i; j++){
                if(dps[j] && wordDictSet.contains(s.substring(j, i))){
                    dps[i] = true;
                    break;
                }
            }
        }
        return dps[s.length()];
    }

    /**
     * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
     * 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
     * 你可以认为每种硬币的数量是无限的。
     * 我们要兑换的总金额是amount， 第i枚硬币的面值 coins[i] xi为面值为coins[i]的硬币数量。
     * 由于对应面值*对应面值的硬币数量不能超过总金额，得出，对应面值硬币的数量最多不会超过 总金额/对应面值
     * 一个简单的方案通过回溯的方法枚举每个硬币数量子集[x0 .... xn-1]针对给定的子集计算他们组成的金额数，如果金额数位s，则记录返回
     * 合法硬币总数的最小值
     * <p></p>
     * 方法1 记忆化搜索。
     * 组成金额所需的最少硬币数量和可选的n枚硬币面额值之间有一个最优的子结构性质。
     * 假设我们直到组成金额S的最少硬币数，而所需的最后一枚硬币的面值为C
     * 则F(S) = F(S-C) + 1
     * 但是我们不知道最后一个硬币的面值是多少，所以我们需要枚举每个硬币的面额值，并且选择其中的最小值
     * 最少硬币书为0，则总金额为0，当不同面额的硬币为0，我们推算出来所需硬币数为-1（公式）
     * F(s) = 0, 当S=0，F(s) = -1, 当n=0
     * 公式化表达是
     * F(S) = min(i=0..n-1)F(S-ci) + 1, 需要S-ci大于等于0
     * 其中由许多子问题被计算了，所以我们需要将每个子问题的答案进行存储，当下次还要计算时，直接从数组中取出返回即可
     *
     */
    public int coinChange(int[] coins, int amount) {
        if(amount < 1) return 0;
        return coinChangeRecur(coins, amount, new int[amount+1]);
    }

    private int coinChangeRecur(int[] coins, int amount, int[] f){
        // 当前总金额 < 0，则所需硬币数为-1
        if(amount < 0) return -1;
        // 当前总金额 等于 0，则所需硬币数为0
        if(amount == 0) return 0;
        // 如果子问题也就是总金额是amount时的最少硬币数不为0，则直接返回该数
        if(f[amount] != 0) return f[amount];
        // 否则我们需要计算子问题所需的最少硬币数，当前子问题需要从coins.length()个选择中选择一个面值最小的
        // 则当前问题所需的最少硬币数就是子问题的最少硬币数 + 1
        int min = Integer.MAX_VALUE;
        for(int coin : coins){
            int result = coinChangeRecur(coins, amount - coin, f);
            if(result>=0 && result<min){
                min = 1+ result;
            }
        }
        f[amount] = min == Integer.MAX_VALUE ? -1 : min;
        return f[amount];
    }

    /**
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     * 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。
     * 例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
     * 定义dp[i]为考虑前i个元素，以第i个数字结尾的最长上升子序列长度，nums[i]必须被选取
     * 我们从小到达计算dp数组的值，在计算dp[i]之前，我们已经计算出dp[0....i-1]的值
     * 则dp[i] = max(dp[j]) + 1，其中j在0和i之间，并且j下标的数字要小于i下标对应的数字
     * 考虑往dp[0...i-1]中的最长上升子序列后面在加一个nums[i]
     * 由于dp[j]所代表的时nums[0...j]中以nums[j]结尾的最长上升子序列，所以，如果能够从dp[j]这个状态
     * 转移过来，那么nums[i]必然要大于nums[j]才能够将nums[i]放在nums[j]后面用来形成更长的上升子序列
     * 最后，整个数组的最长上升子序列即所有dp[i]中的最大值
     */
    public int lengthOfLIS(int[] nums) {
        // 当数组长度为0时
        if (nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        // 只有一个元素的最长子序列长度为1，（在数组下标为0）
        dp[0] = 1;
        int maxAnswer = 1;
        // 从数组的第二个元素开始，循环遍历nums数组，遍历下标获取其子问题的解，
        for(int i = 1; i<nums.length; i++){
            // 超过一个元素的最长子序列长度最小为1
            dp[i] = 1;
            for(int j=0; j<i; j++){
                if(nums[i] > nums[j]){
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            maxAnswer = Math.max(maxAnswer, dp[i]);
        }
        return maxAnswer;
    }

    /**
     * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * 此外，你可以假设该网格的四条边均被水包围。
     *
     * 深度优先搜索
     * 我们可以将二维网格看成一个无向图，竖直或者水平相邻的1之间有边相连
     * 为了求出岛屿的数量，我们可以扫描整个二维网络，如果一个位置是1，则以其为起始节点开始进行深度优先搜索
     * 每个搜索到的1都会被重新标记为0，而搜到0则终止该分支。最后岛屿的数量就是图中存在的1的数量
     *
     * 广度优先搜索
     * 如果一个位置是1，则将其加入队列，开始进行广度有限搜索，在这个过程中，每个搜索到的1都会被重新标记为0，直到队列为空，则搜索结束
     *
     * 并查集
     * 为了求出岛屿的数量，我们可以扫描整个二维网络，如果一个位置是1，则将其与相邻四个方向上的1在并查集中合并
     */
    public int numIslands(char[][] grid) {
        // 边界条件
        if(grid == null || grid.length == 0) return 0;
        int nr = grid.length;
        int nc = grid[0].length;
        int numLands = 0;
        for(int r = 0; r < nr; r++){
            for(int c = 0; c < nc; c++){
                if(grid[r][c] == '1'){
                    numLands++;
                    dfs(grid,r,c);
                }
            }
        }
        return numLands;
    }

    // r, c为当前结点坐标
    private void dfs(char[][] grid, int r, int c){
        int nr = grid.length;
        int nc = grid[0].length;
        // 边界条件，当前节点<0，当前节点坐标大于整个数组的长宽， 当前结点的值为’0‘
        if(r<0||c<0||r>=nr||c>=nc||grid[r][c] == '0') return;
        // 先将当前结点标记为0，然后遍历上下左右的结点
        grid[r][c] = '0';
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }

    /**
     * 给你一个 m * n 的矩阵 board ，由若干字符 'X' 和 'O' ，
     * 找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
     * 矩阵有三种元素：
     * 字母X
     * 被字母X包围的字母O
     * 没有被字母X包围的字母O
     * 要求是将所有被字母X包围的字母O都变成X
     * 而任何边界上的O都不会被填充为X，也就是所有的不被包围的O都直接或者间接与边界上的O相连
     * 对于每一个边界上的O，我们以他为起点，标记所有与他直接或者间接相连的字母O
     * 最后我们遍历这个矩阵，对于每一个字母
     * 如果该字母已经被标记过了，则该字母为没有被字母X包围的字母O，我们将其还原为字母O
     * 如果该字母没有被标记过，则该字母为被字母X包围的字母O，我们将其修改为X
     */
    int n;
    int m;
    public void solve(char[][] board) {
        n = board.length;
        if(n == 0) return;
        m = board[0].length;
        // 以边界上的O为起点，标记与他相连的字母O
        for(int i = 0; i<n;i++){
            // 以矩阵左、下的边界为起点
            dfsSolve(board,i,0);
            dfsSolve(board, i,m-1);
        }
        for(int i = 1; i<m-1; i++){
            // 以矩阵的右、上的边界为起点
            dfsSolve(board, 0, i);
            dfsSolve(board,n-1, i);
        }
        // 遍历整个矩阵
        for(int i=0;i<n;i++){
            for(int j=0; j<m;j++){
                if(board[i][j] == 'A') {
                    board[i][j] = 'O';
                }else {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void dfsSolve(char[][] board, int x, int y){
        if(x<0||x>=n||y<0||y>=m||board[x][y] != 'O') return;
        // 标记当前的O
        board[x][y] = 'A';
        dfsSolve(board,x+1,y);
        dfsSolve(board,x-1,y);
        dfsSolve(board,x,y+1);
        dfsSolve(board,x,y-1);
    }
}
```

```java
package org.pei;

import java.util.ArrayList;
import java.util.List;

public class Node {
    public int val;
    public List<Node> neighbors;
    public Node() {
        this.val = 0;
        this.neighbors = new ArrayList<>();
    }
    public Node(int val) {
        this.val = val;
        this.neighbors = new ArrayList<>();
    }
    public Node(int val, List<Node> neighbors) {
        this.val = val;
        this.neighbors = neighbors;
    }
}

```

```java
package org.pei;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution {

    /**
     * 给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。
     * 对于一张图而言，他的深拷贝即构建一张与原图结构，值均一样的图，但是其中的结点不再是原来图结点的引用，因此，为了深拷贝出整张图
     * 我们需要直到整张图的结构以及对应节点的值。
     * 由于题目给定了我们一个结点的引用，我们为了知道整张图的结构以及对应的值，我们需要从给定的结点出发，进行图的遍历，并且在遍历的过程中
     * 完成图的深拷贝。为了避免在深拷贝中陷入死循环。也就是多次遍历同一个结点，我们需要一种数据结构记录已经被克隆过的结点
     * 1. 使用一个哈希表存储所有已经被访问和克隆的结点，哈希表中的key是原始图中的结点，而value是克隆图中的对应结点
     * 2. 从给定结点开始遍历图，如果某个结点已经被访问过，则返回其克隆图中的对应结点
     * 3. 如果当前访问的结点不再哈希表中，则创建他的克隆结点并且存储在哈希表中，如果不保证这种顺序，可能会在递归中再次遇到同一个结点
     * 再次遍历该节点时，会陷入到死循环
     * 4. 递归调用每个结点的邻接点，每个结点递归调用的次数等于邻接点的数量，每一次调用返回其对应邻接点的克隆结点，最终返回这些克隆邻接点的列表
     * 将其放入在对应克隆结点的邻接表中，这样就可以克隆给党的结点和邻接点
     */
    private HashMap<Node, Node> visitedNode = new HashMap<>();
    public Node cloneGraph(Node node) {
        if(node == null) return null;
        // 如果访问过该节点，则返回对应的克隆节点
        if(visitedNode.containsKey(node)) return visitedNode.get(node);
        // 否则克隆结点，并且放入到哈希表中
        Node cloneNode = new Node(node.val, new ArrayList<>());
        visitedNode.put(node,cloneNode);
        // 由于之前的克隆结点并没有构建结点之间的邻居关系，因此我们再遍历完结点后应该补充构建一下
        // 最后再遍历一遍原图，找到其相邻结点，克隆其相邻结点并建立邻居关系
        for(Node neighbor: node.neighbors){
            cloneNode.neighbors.add(cloneGraph(neighbor));
        }
        return cloneNode;
    }

    /**
     * 给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，
     * 其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。
     * 每个 Ai 或 Bi 是一个表示单个变量的字符串。
     * 另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，
     * 请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。=
     * 返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。
     * 如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。
     * 注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。
     * 注意：未在等式列表中出现的变量是未定义的，因此无法确定它们的答案。
     * <p/>
     * 由于变量之间的倍数关系具有传递性，处理有传递性关系的问题，可以使用并查集
     * 需要再并查集的合并和查询中维护这些变量之间的倍数关系
     * 我们可以把整个问题建模成一张图，给定图中的一些点，以及某些边的权值，指着对任意两点求出其路径长
     * 因此，我们首先需要遍历equations数组，找出其中所有不同的字符串，并且通过哈希表将每个不同的字符串映射成整数
     */
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        return new double[0];
    }

    /**
     *你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
     * 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，
     * 其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
     * 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
     * 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
     * <p/>
     * 给定一个包含n个结点的有向图，我们给出他的结点编号的一种排列，如果满足，对于图G中的任意一条有向边，
     * u再排列中都出现在v的前面
     * 那么称该排列时图G的拓扑排序，根据上述的定义，我们可以得出
     * 1. 如果图中存在环，也就是图不是有向无环图，那么图就不存在拓扑排序
     * 2. 如果图是一个邮箱无环图，那么他的拓扑排序可能不止一种
     * <p></p>
     * 对于本题，我们将每一门课看成一个结点，如果想要学习课程A之前必须完成课程B，那么我们从B到A需要连接一条有向边
     * 这样以来，在拓扑排序中B一定在A的前面
     * 方法1：深度优先搜索
     * 我们可以将深度优先搜索的流程和拓扑排序的求解联系起来，用一个栈来存储所有已经搜索完成的结点，对于一个结点，如果
     * 他的所有相邻结点都已经搜索完，那么在搜索回溯到u的时候，u本身也会变成一个已经索索完成的结点，这里的相邻结点
     * 值得时从u出发，通过一条有向边就能到达的所有结点。
     * 假设我们当前搜索到了结点u，如果他所有相邻结点都已经搜索完成，那么这些结点都已经在栈中了，此时我们可以将u入栈，
     * 如果我们从栈顶往栈底的顺序来看，由于u处于栈顶的位置，那么u出现在所有u的相邻结点前面，因此对于u这个结点而言，他是
     * 满足拓扑排序要求的，这样以来，我们对图进行一遍深度有限搜索，当每个结点进行回溯的时候，我们把该节点放入栈中，最
     * 终从栈顶到栈底的序列就是一种拓扑排序
     * <p/>
     * 对于图中的任意结点，它在搜索的过程中，有三种状态
     * 1. 未搜索，我们还没有搜索到该节点
     * 2. 搜索中，我们搜索过该节点，但是还没有回溯到该节点，即该节点还没有入栈，还有相邻的结点没有搜索完成、
     * 3. 已完成，我们搜索并回溯过这个结点，也就是说这个结点已经入栈，并且所有该节点的相邻结点都已经出现在栈的更底部的位置，满足
     * 拓扑排序的要求
     *
     * 因此，我们在每一轮搜索开始时，任取一个未搜索的结点开始进行深度优先搜索
     * 将当前搜索的结点u标记为搜索中，遍历该节点的每一个相邻结点v
     * 1， 如果v是未搜索状态，那么我们开始搜索v，等到搜索完成之后回溯到u
     * 2. 如果v是搜索中，那么我们就找到图中存在的环，因此是不存在拓扑排序的
     * 3. 如果v是已完成，说明v已经在栈中，但是u还不在，因此u无论何时入栈都不会 影响到uv之间的拓扑关系
     * 当u的所有相邻结点都为已完成时，我们将u放入栈中，并且将其标记为已完成
     * 在整个深度有限搜索过程结束之后，如果我们没有找到图中的环，那么栈中存储的所有的n个结点，从栈顶一直到栈底的顺序就是一种拓扑排序
     *
     */
    List<List<Integer>> edges;
    // 0 未访问 1 访问中 2 已完成
    int[] visited;
    // 标记当前是否能够完成所有课程的学习
    boolean valid = true;
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // 新建一个图的数据结构
        edges = new ArrayList<>();
        // 将所有的课程作为结点都加入图中
        for(int i = 0; i<numCourses; i++){
            edges.add(new ArrayList<>());
        }
        // 初始化每个节点的访问状态
        visited = new int[numCourses];
        // 为课程添加先修课程
        for(int[] info : prerequisites){
            edges.get(info[1]).add(info[0]);
        }
        // 遍历访问每一个结点，构建一个拓扑排序
        for(int i=0; i<numCourses && valid; i++){
            if(visited[i] == 0) dfs(i);
        }
        return valid;
    }
    public void dfs(int u){
        // 将当前结点状态标记为访问中
        visited[u] = 1;
        // 获取其所有的相邻结点，判断其访问状态
        // 1， 如果v是未搜索状态，那么我们开始搜索v，等到搜索完成之后回溯到u
        // 2. 如果v是搜索中，那么我们就找到图中存在的环，因此是不存在拓扑排序的
        // 3. 如果v是已完成，说明v已经在栈中，但是u还不在，因此u无论何时入栈都不会 影响到uv之间的拓扑关系
        // 当u的所有相邻结点都为已完成时，我们将u放入栈中，并且将其标记为已完成
        for(int v: edges.get(u)){
            if(visited[v] == 0){
                dfs(v);
                if(!valid) return;
            } else if (visited[v] == 1) {
                valid = false;
                return;
            }
        }
        visited[u] = 2;
    }

    /**
     * 给定一个三角形 triangle ，找出自顶向下的最小路径和。
     * 每一步只能移动到下一行中相邻的结点上。
     * 相邻的结点 在这里指的是 下标 与 上一层结点下标 相同
     * 或者等于 上一层结点下标 + 1 的两个结点。
     * 也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        return 0;
    }

    /**
     * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
     * 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
     * 回溯法：
     * 首先，使用哈希表存储每个数字对应的所有可能的字母，然后进行回溯操作
     * 回溯过程中维护一个字符串，表示已有的字母排列，如果没有遍历完电话号码的所有数字，则已有的字母排列是不够完整的
     * 该字符串初始为空，每次取电话号码的一位数字，从哈希表中获得该数字对应的所有可能的字母，并且将其中的一个字母插入到已有的字母排列
     * 后面，然后继续处理电话号码的后一位数字，直到处理电话号码的所有数字，即能够得到一个完整的字母序列，然后进行回退操作
     */
    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<>();
        if (digits.length() == 0){
            return combinations;
        }
        Map<Character, String> phoneMap = new HashMap<>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        backTrack(combinations, phoneMap, digits, 0, new StringBuffer());
        return combinations;
    }

    private void backTrack(List<String> combinations, Map<Character, String> phoneMap,
                           String digits, int index, StringBuffer combination) {
        if(digits.length() == index){
            combinations.add(combination.toString());
        }else {
            char digit = digits.charAt(index);
            String letters = phoneMap.get(digit);
            int letterCount = letters.length();
            for(int i = 0; i<letterCount; i++){
                combination.append(letters.charAt(i));
                backTrack(combinations,phoneMap,digits,index+1, combination);
                combination.deleteCharAt(index);
            }
        }
    }

    /**
     * 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
     * 你可以按 任何顺序 返回答案。
     * 从
     */
    public List<List<Integer>> combine(int n, int k) {
        return null;
    }

     public class TreeNode {
         int val;
         TreeNode left;
         TreeNode right;
         TreeNode() {}
         TreeNode(int val) { this.val = val; }
         TreeNode(int val, TreeNode left, TreeNode right) {
             this.val = val;
             this.left = left;
             this.right = right;
         }
     }

    /**
     * 给定二叉树，返回其最大深度
     * 如果我们已经知道了左右子树的最大深度，那么当前二叉树的最大深度就是左右子树中最大的深度+1
     * 而左右子树的最大深度可以用同样的方式计算，因此我们可以使用深度优先搜索的方法计算二叉树最大深度
     *
     */
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        int leftHeight = maxDepth(root.left);
        int rightHeight = maxDepth(root.right);
        return Math.max(leftHeight, rightHeight) + 1;
    }

    /**
     * 两个二叉树相同，当且仅当两个二叉树的结构完全相同，且所有对应结点的值相同，因此可以通过搜索的方式判断
     * 深度优先搜索
     * 如果两个二叉树都为空，则两个二叉树相同，如果两个二叉树只有一个为空，则一定不相同
     * 如果两个二叉树都不空，首先判断其根节点的值是否相同，然后在分别判断其左右子树是否相同
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if (p == null || q == null) return false;
        if(p.val != q.val) return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    /**
     * 先从根节点开始
     */
    public TreeNode invertTree(TreeNode root) {
        return null;
    }


    static boolean a;

    public static void main(String[] args) {
        System.out.println(a);
    }

}

```
