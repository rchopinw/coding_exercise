import heapq


class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val
        self.next = nxt


# merging k sorted arrays
def merge_k_sorted_arrays(arrays):
    ans, heap = [], []
    for i, array in enumerate(arrays):
        if array:
            heapq.heappush(heap, (array[0], i, 0))
    while heap:
        num, array_idx, item_idx = heapq.heappop(heap)
        ans.append(num)
        if item_idx < len(arrays[array_idx]) - 1:
            heapq.heappush(heap, (arrays[array_idx][item_idx + 1], array_idx, item_idx + 1))
    return ans


def merge_k_sorted_lists(lists):
    pq = []
    m = ListNode()
    cur = m
    for i, l in enumerate(lists):
        if l:
            heapq.heappush(pq, (l.val, i))
            lists[i] = lists[i].next
    while pq:
        cur_val, f = heapq.heappop(pq)
        cur.next = ListNode(val=cur_val)
        if lists[f]:
            heapq.heappush(pq, (lists[f].val, f))
            lists[f] = lists[f].next
        cur = cur.next
    return m.next


if __name__ == '__main__':
    arrs = [[2, 5, 8, 10], [3, 7], [2, 3, 22], [1]]
    assert merge_k_sorted_arrays(arrs) == [1, 2, 2, 3, 3, 5, 7, 8, 10, 22]

    arrs = [[], [1, 2, 5], [3]]
    assert merge_k_sorted_arrays((arrs)) == [1, 2, 3, 5]
    