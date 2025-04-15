import sys

def longest_unique_substring(s):
    start = 0
    max_len = 0
    max_substring = ""
    seen = {}

    for end, char in enumerate(s):
        if char in seen and seen[char] >= start:
            start = seen[char] + 1
        seen[char] = end
        if end - start + 1 > max_len:
            max_len = end - start + 1
            max_substring = s[start:end+1]

    return max_substring

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Length of the string should be bigger than 2")
        sys.exit(1)
    
    input_string = sys.argv[1]
    result = longest_unique_substring(input_string)
    print("Longest substring without duplicate characters:", result)
    print("time complexity: O(len(input_string)) - space complexity: O(input_string)")
