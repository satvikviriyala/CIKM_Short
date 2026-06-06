import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Set up figure
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_aspect('equal')
ax.axis('off')

# Equilateral triangle vertices
# A: Top (Correctness), B: Bottom Left (Non-Bias), C: Bottom Right (Utility)
A = np.array([0.5, np.sqrt(3)/2])
B = np.array([0, 0])
C = np.array([1, 0])

# Draw triangle
triangle = patches.Polygon([A, B, C], closed=True, fill=False, edgecolor='gray', linewidth=3, zorder=1)
ax.add_patch(triangle)

# Vertex labels
plt.text(A[0], A[1] + 0.05, "Strong Correctness", ha='center', va='bottom', fontsize=16, fontweight='bold')
plt.text(B[0] - 0.05, B[1] - 0.05, "Strict Non-Bias", ha='right', va='top', fontsize=16, fontweight='bold')
plt.text(C[0] + 0.05, C[1] - 0.05, "Decisive Utility", ha='left', va='top', fontsize=16, fontweight='bold')

# Helper to convert weights to Cartesian coordinates
def get_cartesian(w_c, w_nb, w_u):
    total = w_c + w_nb + w_u
    w = np.array([w_c, w_nb, w_u]) / total
    return w[0]*A + w[1]*B + w[2]*C

# Data mapping
# We use actual data to derive visual weights so the positions reflect the empirical trade-off.
# Correctness: % Corr (scaled to balance visual weight)
# Non-Bias: Inversely proportional to CSS (0.040 - CSS) * scale
# Utility: % Decisive
raw_data = {
    "Vanilla": {"C": 4.2, "CSS": 0.034, "U": 1.30, "color": "#1f77b4"},
    "Utility-First": {"C": 4.4, "CSS": 0.036, "U": 3.28, "color": "#ff7f0e"},
    "Neutrality": {"C": 5.6, "CSS": 0.032, "U": 0.34, "color": "#2ca02c"},
    "Clarification-First": {"C": 3.5, "CSS": 0.030, "U": 0.24, "color": "#9467bd"}
}

points = {}
for name, d in raw_data.items():
    # Derive weights to position them meaningfully within the triangle
    w_c = d["C"] * 6           # Scale up correctness so points aren't purely at the bottom
    w_nb = (0.040 - d["CSS"]) * 3000 # .030 -> 30, .036 -> 12
    w_u = d["U"] * 30          # 0.24 to 3.28 scaled up to balance weights
    
    pos = get_cartesian(w_c, w_nb, w_u)
    
    # Plot point
    ax.scatter(pos[0], pos[1], s=400, color=d["color"], edgecolor='white', linewidth=2, zorder=3)
    
    # Add label
    # Adjust label position slightly based on point location
    if name == "Utility-First":
        plt.text(pos[0], pos[1] - 0.06, name, ha='center', va='top', fontsize=12, fontweight='bold')
    elif name == "Clarification-First":
        plt.text(pos[0] - 0.04, pos[1], name, ha='right', va='center', fontsize=12, fontweight='bold')
    elif name == "Neutrality":
        plt.text(pos[0] - 0.04, pos[1], name, ha='right', va='center', fontsize=12, fontweight='bold')
    else:
        plt.text(pos[0] + 0.04, pos[1], name, ha='left', va='center', fontsize=12, fontweight='bold')

plt.title("The Correctness-Non-Bias-Utility (C-NB-U) Trilemma", fontsize=18, y=1.05)
plt.figtext(0.5, -0.05, "Points mapped empirically from % Strong Correctness, CSS, and Decisive Utility.", 
            ha="center", fontsize=11, color="dimgray")

# Save as PDF and PNG for LaTeX
plt.savefig("Paper Writing/trilemma.pdf", bbox_inches='tight')
plt.savefig("Paper Writing/trilemma.png", bbox_inches='tight', dpi=300)
plt.close()
print("Saved trilemma.pdf and trilemma.png")
