import re

def fix_commas(text: str) -> str:
    text = re.sub(r'(?:(?<=^)|(?<="))",', '', text)
    # text = re.sub(r'""([^"]+)"', r'""\1""', text)
    # text = re.sub(r'""([^"]*?)"(?!")', r'""\1""', text)
    return text

with open("data/test.csv", "r", encoding="utf-8") as f:
    content = f.read()

fixed = fix_commas(content)

with open("data/test_fixed.csv", "w", encoding="utf-8") as f:
    f.write(fixed)
    
    
print("completed")