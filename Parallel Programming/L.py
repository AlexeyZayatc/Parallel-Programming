s = list(map(int, input()))

ans = 0
arr = {i: -1 for i in range(10)}
for i, l in enumerate(s):
    if arr[l] != -1:
        ans += i - arr[l] - 1
        arr[l] += 1
    else:
        k = -1
        for j in range(l + 1, 10):
            if arr[j] != -1:
                k = arr[j]
                break
        if k != -1:
            ans += i - k - 1
            arr[l] = k + 1
        else:
            ans += i
            arr[l] = 0
    for j in range(0, l):
        if arr[j] != -1:
            arr[j] += 1
print(ans)