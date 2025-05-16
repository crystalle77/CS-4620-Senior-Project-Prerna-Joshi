
import matplotlib.pyplot as plt
import numpy as np

# Disease labels
diseases = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", 
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", 
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]

# AUC scores
same_race_same = [0.6294, 0.8069, 0.7014, 0.6492, 0.7988, 0.6238, 0.6690, 0.6155, 0.8210, 0.8343, 0.7369, 0.6160, 0.8071, 0.8429]
same_race_diff = [0.6168, 0.8016, 0.6707, 0.6314, 0.7767, 0.6268, 0.6752, 0.6202, 0.7853, 0.8291, 0.7093, 0.6235, 0.8223, 0.8276]
diff_race_same = [0.6039, 0.8051, 0.6829, 0.6452, 0.7757, 0.6482, 0.6385, 0.6142, 0.8042, 0.8327, 0.6939, 0.6422, 0.8221, 0.8305]
diff_race_diff = [0.6194, 0.8210, 0.6621, 0.6560, 0.7731, 0.6580, 0.6671, 0.6094, 0.8135, 0.8386, 0.6781, 0.6342, 0.8379, 0.8361]

# X locations
x = np.arange(len(diseases))
width = 0.2

# Plotting
plt.figure(figsize=(16, 6))
plt.bar(x - 1.5*width, same_race_same, width, label='Same-Race Model on Same-Race Test', color='skyblue')
plt.bar(x - 0.5*width, same_race_diff, width, label='Same-Race Model on Diff-Race Test', color='deepskyblue')
plt.bar(x + 0.5*width, diff_race_same, width, label='Diff-Race Model on Same-Race Test', color='lightgreen')
plt.bar(x + 1.5*width, diff_race_diff, width, label='Diff-Race Model on Diff-Race Test', color='seagreen')

# Aesthetics
plt.ylabel('AUC Score')
plt.xlabel('Disease')
plt.title('Model Performance Across Disease Labels and Racial Groups')
plt.xticks(x, diseases, rotation=45, ha='right')
plt.ylim(0.5, 0.9)
plt.legend(loc='lower right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


plt.savefig("AUCScore.png")
