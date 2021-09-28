import heapq


# merging k sorted arrays
def merge_k_sorted_arrays(arrays):
    ans, heap = [], []
    for i, array in enumerate(arrays):
        heapq.heappush(heap, (array[0], i, 0))
    while heap:
        num, array_idx, item_idx = heapq.heappop(heap)
        ans.append(num)
        if item_idx < len(arrays[array_idx]) - 1:
            heapq.heappush(heap, (arrays[array_idx][item_idx + 1], array_idx, item_idx + 1))
    return ans

