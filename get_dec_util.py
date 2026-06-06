import re

tex = open("Paper Writing/main.tex").read()

# Find Table 3
start = tex.find(r"\multirow{4}{*}{Gemma 4 E4B}")
end = tex.find(r"\end{tabular*}", start)
table3 = tex[start:end]

lines = table3.split("\n")
new_lines = []
for line in lines:
    if "&" in line and "\\multirow" not in line and "midrule" not in line:
        parts = line.split("&")
        if len(parts) >= 10:
            util = float(re.sub(r'[^0-9.]', '', parts[2]))
            dec = float(re.sub(r'[^0-9.]', '', parts[6]))
            dec_util = util * (dec / 100.0)
            
            # Format dec_util with 2 decimal places
            # Insert it after Utility
            
            # Reconstruct line
            # Wait, better to insert after Utility or at the end? Let's insert after Utility
            # parts[2] is Utility. Let's insert dec_util before Satisf.
            parts.insert(3, f" {dec_util:.2f} ")
            new_lines.append("&".join(parts))
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

print("\n".join(new_lines[:15]))
