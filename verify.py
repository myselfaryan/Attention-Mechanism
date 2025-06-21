from sys import argv

assert len(argv) >= 3, "python verify.py FILE1 FILE2"

with open(argv[1]) as f1, open(argv[2]) as f2:
    dims1 = tuple(map(int, next(f1).split()))
    dims2 = tuple(map(int, next(f2).split()))
    assert dims1 == dims2, (
        f"Dims {dims1} and {dims2} don't match between FILE1 and FILE2"
    )

    m, n = dims1
    for i in range(m):
        s1 = tuple(map(float, next(f1).split()))
        assert len(s1) == n, "Line in FILE1 is smaller than promised"

        s2 = tuple(map(float, next(f2).split()))
        assert len(s2) == n, "Line in FILE2 is smaller than promised"

        assert all(abs(a - b) <= 1e-3 for a, b in zip(s1, s2)), (
            "Abs diff more than 1e-3"
        )
