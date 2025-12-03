
# app.py
import sys, platform, pathlib, statistics, random

def summarize(nums):
    return {
        "count": len(nums),
        "mean": statistics.mean(nums),
        "stdev": statistics.pstdev(nums) if len(nums) > 1 else 0.0,
    }

if __name__ == "__main__":
    print(f"Hello from Python {platform.python_version()} @ {sys.executable}")
    nums = [random.randint(1, 100) for _ in range(10)]
    print("Sample numbers:", nums)
    print("Summary:", summarize(nums))
    out = pathlib.Path("test_output.txt")
    out.write_text("Python in Visual Studio is working.\n")
    print("Wrote:", out.resolve())