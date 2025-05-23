import os
import re

def process_literalinclude(path, language):
    # Make the path relative to the current directory if it starts with ../
    if path.startswith("../"):
        path = path[3:]  # Remove the ../
    
    if os.path.exists(path):
        with open(path, "r") as code_file:
            code = code_file.read()
            return f".. code-block:: {language}\n\n{code}\n\n"
    else:
        return f".. warning:: File {path} not found\n\n"

res = ""
with open("README_template.rst", "r") as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip("\n")
        i += 1
        
        # Handle include directives
        if line.startswith(".. include:: "):
            include_path = line.replace(".. include:: ", "")
            if include_path.endswith(".rst"):
                with open(include_path, "r") as tf:
                    content = tf.read()
                    # Process literalinclude directives inside the included file
                    content = re.sub(r'\.\. literalinclude:: (.*?)\n\s+:language: (.*?)\n', 
                        lambda m: process_literalinclude(m.group(1), m.group(2)), content)
                    res += content.replace(".. literalinclude:: ../LICENSE", "").replace("<../LICENSE>", "<LICENSE>").replace(":class:", " ")
        # Handle literalinclude directives directly in the template
        elif line.startswith(".. literalinclude:: "):
            # Extract path and language
            path_match = re.match(r'\.\. literalinclude:: (.*)', line)
            if path_match:
                include_path = path_match.group(1)
                # Read the next line to get the language if available
                language = "python"  # Default language
                if i < len(lines) and lines[i].strip().startswith(":language:"):
                    language = lines[i].strip().replace(":language:", "").strip()
                    i += 1  # Skip the language line since we've processed it
                
                res += process_literalinclude(include_path, language)
        else:
            res += line + "\n"

print(res)

with open("README.rst", "w+") as f:
    f.write(res)
