from utils import get_limit

def run_task():
    """
    Runs a task based on the limit retrieved from utils.
    Returns a list of numbers if limit is valid, otherwise None.
    """
    limit = get_limit()
    print(f"DEBUG: limit is {limit}")
    if limit < 1:
        # Unexpected: why is limit 0?
        return None
    return [i for i in range(limit)]

if __name__ == "__main__":
    result = run_task()
    print(f"Result: {result}")
