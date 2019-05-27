class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        len1, len2 = len(nums1), len(nums2)
        if len1 and len2:
            list_sum = []
            a, b = 0, 0
            min1, min2 = nums1[a], nums2[b]
            while True:
                if min1 < min2:
                    list_sum.append(min1)
                    a += 1
                    if a - len1:
                        min1 = nums1[a]
                    else:
                        list_sum.extend(nums2[b:])
                        break
                else:
                    list_sum.append(min2)
                    b += 1
                    if b - len2:
                        min2 = nums2[b]
                    else:
                        list_sum.extend(nums1[a:])
                        break
            length = len1+len2
        elif not nums1:
            list_sum = nums2
            length = len2
        else:
            return nums1
            length = len1
        if length % 2:
            return list_sum[length//2]
        else:
            return (list_sum[length//2-1] + list_sum[length//2])/2
    
    def sort_two_list(self, nums1, nums2):
        